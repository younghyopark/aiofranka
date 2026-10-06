// aiofranka._native: the 1 kHz control loop of NativeFrankaController, in C++.
//
// The loop runs in its own thread and never touches Python. Each cycle it reads the robot
// state through the ActiveControl that pylibfranka started, computes the flange pose,
// Jacobian and mass matrix with the libmujoco that the mujoco package loaded, runs the
// control law, and sends the torques. Python writes the controller's attributes into a
// staging copy of Params, which the loop copies at the start of each cycle, and reads what
// the loop did from a snapshot. Neither side waits for the other: the loop only try-locks.
// A recording takes chosen fields of every cycle into a ring that Python drains.
//
// libfranka is not linked. aiofranka.native loads pylibfranka's copy with RTLD_GLOBAL
// before importing this module, so the vendored headers (include/franka) must be those of
// pylibfranka's libfranka version.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/conduit/pybind11_conduit_v1.h>

#include <franka/active_control_base.h>
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot_state.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <pthread.h>
#if defined(__APPLE__)
#include <sys/qos.h>
#else
#include <sched.h>
#endif

#include "laws.h"

namespace py = pybind11;

namespace aiofranka {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kLibfrankaVersion = "0.21.3";  // of the headers in include/franka
constexpr int kMaxLaws = 64;

// The MuJoCo functions the loop calls, from the mujoco package's libmujoco.
struct MjApi {
  void (*fwd_position)(const void*, void*) = nullptr;
  void (*step)(const void*, void*) = nullptr;
  void (*jac_site)(const void*, const void*, double*, double*, int) = nullptr;
  void (*full_m)(const void*, const void*, double*) = nullptr;
};

// One mjData of a model with nq = nv = nu = 7, by the addresses of its arrays.
struct MjArrays {
  void* data = nullptr;
  double* qpos = nullptr;
  double* qvel = nullptr;
  double* ctrl = nullptr;
  const double* site_xpos = nullptr;  // of the flange site
  const double* site_xmat = nullptr;
};

struct Realtime {
  bool macos_qos = true;
  int cpu = -1;
  int fifo_priority = 0;
};

// Timing of the cycles. Window fields cover the time since the last read with reset; warn,
// error and max_all cover the time since start.
struct Stats {
  // Periods between cycle starts [s].
  int64_t count = 0;
  double sum = 0, sum_sq = 0, min = 0, max = 0;
  // Time from a cycle's start to its torque command [s].
  int64_t busy_count = 0;
  double busy_sum = 0, busy_max = 0;
  // What the robot reported (real robot only): the largest gap between two states it sent
  // [s], the states missed in between, and the lowest share of commands that reached it.
  double robot_gap_max = 0;
  int64_t missed = 0;
  double success_min = 1.0;
  // Periods off 1 ms by more than 0.1 ms, and longer than 10 ms.
  int64_t warn = 0, error = 0;
  double max_all = 0;

  void add(double dt) {
    min = count == 0 ? dt : std::min(min, dt);
    max = count == 0 ? dt : std::max(max, dt);
    ++count;
    sum += dt;
    sum_sq += dt * dt;
    max_all = std::max(max_all, dt);
    if (dt > 10e-3) {
      ++error;
    } else if (dt < 0.9e-3 || dt > 1.1e-3) {
      ++warn;
    }
  }

  void add_busy(double busy) {
    ++busy_count;
    busy_sum += busy;
    busy_max = std::max(busy_max, busy);
  }

  void add_robot(double gap, double success) {
    robot_gap_max = std::max(robot_gap_max, gap);
    if (gap > 1.5e-3) {
      missed += static_cast<int64_t>(std::llround(gap * 1e3)) - 1;
    }
    success_min = std::min(success_min, success);
  }

  void merge(const Stats& other) {
    if (other.count > 0) {
      min = count == 0 ? other.min : std::min(min, other.min);
      max = count == 0 ? other.max : std::max(max, other.max);
    }
    count += other.count;
    sum += other.sum;
    sum_sq += other.sum_sq;
    busy_count += other.busy_count;
    busy_sum += other.busy_sum;
    busy_max = std::max(busy_max, other.busy_max);
    robot_gap_max = std::max(robot_gap_max, other.robot_gap_max);
    missed += other.missed;
    success_min = std::min(success_min, other.success_min);
    warn += other.warn;
    error += other.error;
    max_all = std::max(max_all, other.max_all);
  }

  void reset_window() {
    count = busy_count = missed = 0;
    sum = sum_sq = min = max = busy_sum = busy_max = robot_gap_max = 0;
    success_min = 1.0;
  }
};

struct Snapshot {
  int64_t cycle = 0;
  int64_t mode = 0;
  LawState state{};
  double torque[7]{};  // of the last impedance or PID cycle, FrankaController.torque
  double last_command[7]{};
  double error_integral[7]{};
  double world_qpos[7]{}, world_qvel[7]{}, world_ctrl[7]{};
  double world_time = 0;
  int64_t model_epoch = 0;
  int64_t memory_size = 0;
  double memory[kMemory]{};
};

struct CustomLaw {
  std::string name;
  LawFn fn = nullptr;
  int64_t memory_size = 0;
};

// What a recording can take of a cycle besides the law's state, the controller's attributes
// and the robot state.
struct RecordExtra {
  double wall_time = 0;     // host clock, as time.time(), when the cycle's state arrived [s]
  double busy = 0;          // from the state's arrival to the torque command [s]
  int64_t robot_mode = -1;  // franka::RobotMode, -1 in simulation
  double tau[7]{};          // the torque sent, after the rate limit and clip [Nm]
  double tcp[16]{};         // the flange pose times control_transform
};

// Where a recorded field comes from.
enum Source : int { kFromState = 0, kFromParams = 1, kFromRobot = 2, kFromExtra = 3, kSources = 4 };

std::size_t source_size(int source) {
  switch (source) {
    case kFromState:
      return sizeof(LawState);
    case kFromParams:
      return sizeof(Params);
    case kFromRobot:
      return sizeof(franka::RobotState);
    case kFromExtra:
      return sizeof(RecordExtra);
    default:
      return 0;
  }
}

// A recording: chosen fields of every cycle, in a ring with one producer, the loop, which
// never waits, and one consumer, drain(). When the ring is full, the loop drops the row and
// counts it.
class Recorder {
 public:
  struct Slice {
    int source;
    std::size_t offset;
    std::size_t bytes;
  };

  Recorder(std::vector<Slice> slices, int64_t capacity)
      : slices_(std::move(slices)), capacity_(capacity) {
    if (capacity_ < 1) {
      throw std::invalid_argument("a recording needs room for at least one row");
    }
    for (const Slice& s : slices_) {
      const std::size_t size = source_size(s.source);
      if (size == 0 || s.offset % 8 != 0 || s.bytes % 8 != 0 || s.bytes > size ||
          s.offset > size - s.bytes) {
        throw std::out_of_range("no such field to record");
      }
      row_bytes_ += s.bytes;
    }
    if (row_bytes_ == 0) {
      throw std::invalid_argument("a recording needs at least one field");
    }
    // Writes every page now, so that the loop never faults one in.
    buffer_.assign(static_cast<std::size_t>(capacity_) * row_bytes_, 0);
  }

  // The loop: one row from the sources. A missing source (the robot state, in simulation)
  // gives NaN.
  void push(const void* const sources[kSources]) {
    const int64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_.load(std::memory_order_acquire) >= capacity_) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    uint8_t* row = buffer_.data() + static_cast<std::size_t>(head % capacity_) * row_bytes_;
    for (const Slice& s : slices_) {
      const auto* from = static_cast<const uint8_t*>(sources[s.source]);
      if (from != nullptr) {
        std::memcpy(row, from + s.offset, s.bytes);
      } else {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t i = 0; i < s.bytes; i += sizeof(double)) {
          std::memcpy(row + i, &nan, sizeof(double));
        }
      }
      row += s.bytes;
    }
    head_.store(head + 1, std::memory_order_release);
  }

  // Python: the rows since the last drain, as bytes.
  py::array_t<uint8_t> drain() {
    std::lock_guard<std::mutex> lock(drain_mutex_);
    const int64_t tail = tail_.load(std::memory_order_relaxed);
    const int64_t head = head_.load(std::memory_order_acquire);
    const auto rows = static_cast<std::size_t>(head - tail);
    py::array_t<uint8_t> out(static_cast<py::ssize_t>(rows * row_bytes_));
    const auto first = static_cast<std::size_t>(tail % capacity_);
    const std::size_t before_wrap = std::min(rows, static_cast<std::size_t>(capacity_) - first);
    uint8_t* to = out.mutable_data();
    std::memcpy(to, buffer_.data() + first * row_bytes_, before_wrap * row_bytes_);
    std::memcpy(to + before_wrap * row_bytes_, buffer_.data(), (rows - before_wrap) * row_bytes_);
    tail_.store(head, std::memory_order_release);
    return out;
  }

  std::size_t row_bytes() const { return row_bytes_; }
  int64_t capacity() const { return capacity_; }
  int64_t rows() const { return head_.load(std::memory_order_acquire); }
  int64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

 private:
  std::vector<Slice> slices_;
  int64_t capacity_;
  std::size_t row_bytes_ = 0;
  std::vector<uint8_t> buffer_;
  alignas(64) std::atomic<int64_t> head_{0};  // rows written, by the loop
  alignas(64) std::atomic<int64_t> tail_{0};  // rows drained, by Python
  std::atomic<int64_t> dropped_{0};
  std::mutex drain_mutex_;
};

std::string apply_realtime(const Realtime& rt) {
  std::string status;
#if defined(__APPLE__)
  // libfranka busy-waits for robot states on macOS, which keeps up only on a performance core.
  if (rt.macos_qos && pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) {
    status += "could not set the QoS class USER_INTERACTIVE; ";
  }
#else
  if (rt.cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(rt.cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      status += "could not pin to CPU " + std::to_string(rt.cpu) + "; ";
    }
  }
  if (rt.fifo_priority > 0) {
    sched_param param{};
    param.sched_priority = rt.fifo_priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
      status += "could not set SCHED_FIFO priority " + std::to_string(rt.fifo_priority) +
                " (needs CAP_SYS_NICE); ";
    }
  }
#endif
  return status;
}

template <typename T>
T* address(std::uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

double* writable_array(py::array array, const char* name) {
  if (!array.writeable() || array.size() != kJoints || array.itemsize() != sizeof(double) ||
      !py::isinstance<py::array_t<double>>(array) ||
      !(array.flags() & py::array::c_style)) {
    throw std::invalid_argument(std::string(name) + " must be a writable float64 array of 7");
  }
  return static_cast<double*>(array.mutable_data());
}

// A robot for tests: an ActiveControl that simulates the arm in MuJoCo at 1 kHz, as the
// real robot would answer readOnce() and writeOnce().
class FakeActiveControl : public franka::ActiveControlBase {
 public:
  FakeActiveControl(std::uintptr_t model, MjArrays world, std::uintptr_t step, double period,
                    int64_t fail_after, std::string fail_message)
      : model_(address<const void>(model)),
        world_(world),
        step_(reinterpret_cast<void (*)(const void*, void*)>(step)),
        period_(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period))),
        fail_after_(fail_after),
        fail_message_(std::move(fail_message)) {}

  std::pair<franka::RobotState, franka::Duration> readOnce() override {
    if (!started_) {
      next_ = Clock::now();
      started_ = true;
    }
    next_ += period_;
    std::this_thread::sleep_until(next_);
    if (fail_after_ >= 0 && reads_ >= fail_after_) {
      throw std::runtime_error(fail_message_);
    }
    ++reads_;
    franka::RobotState state{};
    std::copy(world_.qpos, world_.qpos + kJoints, state.q.begin());
    std::copy(world_.qvel, world_.qvel + kJoints, state.dq.begin());
    state.q_d = state.q;
    state.tau_J_d = last_command_;
    state.tau_J = last_command_;
    state.robot_mode = franka::RobotMode::kMove;
    state.control_command_success_rate = 1.0;
    state.time = franka::Duration(static_cast<uint64_t>(reads_));
    return {state, franka::Duration(1)};
  }

  void writeOnce(const franka::Torques& torques) override {
    const Clock::time_point now = Clock::now();
    if (writes_ > 0) {
      const int64_t gap = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_write_).count();
      if (gap > max_gap_ns_.load(std::memory_order_relaxed)) {
        max_gap_ns_.store(gap, std::memory_order_relaxed);
      }
    }
    last_write_ = now;
    last_command_ = torques.tau_J;
    std::copy(torques.tau_J.begin(), torques.tau_J.end(), world_.ctrl);
    step_(model_, world_.data);
    ++writes_;
  }

  void writeOnce(const franka::JointPositions&, const std::optional<const franka::Torques>&) override {
    unsupported();
  }
  void writeOnce(const franka::JointVelocities&, const std::optional<const franka::Torques>&) override {
    unsupported();
  }
  void writeOnce(const franka::CartesianPose&, const std::optional<const franka::Torques>&) override {
    unsupported();
  }
  void writeOnce(const franka::CartesianVelocities&,
                 const std::optional<const franka::Torques>&) override {
    unsupported();
  }
  void writeOnce(const franka::JointPositions&) override { unsupported(); }
  void writeOnce(const franka::JointVelocities&) override { unsupported(); }
  void writeOnce(const franka::CartesianPose&) override { unsupported(); }
  void writeOnce(const franka::CartesianVelocities&) override { unsupported(); }

  int64_t reads() const { return reads_; }
  int64_t writes() const { return writes_; }
  // The longest time between two commands [s], which the robot would see.
  double max_gap() const { return max_gap_ns_.load(std::memory_order_relaxed) * 1e-9; }
  void reset_max_gap() { max_gap_ns_.store(0, std::memory_order_relaxed); }

 private:
  [[noreturn]] static void unsupported() {
    throw std::logic_error("_FakeActiveControl only takes torques");
  }

  const void* model_;
  MjArrays world_;
  void (*step_)(const void*, void*);
  Clock::duration period_;
  int64_t fail_after_;
  std::string fail_message_;
  bool started_ = false;
  Clock::time_point next_{};
  int64_t reads_ = 0;
  int64_t writes_ = 0;
  Clock::time_point last_write_{};
  std::atomic<int64_t> max_gap_ns_{0};
  std::array<double, 7> last_command_{};
};

// Whether pylibfranka's classes are registered in the pybind11 internals this module uses,
// which it shares only when both were built with the same internals version.
bool pylibfranka_types() {
  return py::detail::get_type_info(typeid(franka::RobotState)) != nullptr &&
         py::detail::get_type_info(typeid(franka::ActiveControlBase)) != nullptr;
}

franka::ActiveControlBase* active_control_pointer(py::handle object) {
  if (object.is_none()) {
    throw std::runtime_error("torque control is not running (robot.torque_controller is None)");
  }
  if (py::isinstance<FakeActiveControl>(object)) {
    return object.cast<FakeActiveControl*>();
  }
  // pylibfranka's ActiveControlBase, through the pybind11 internals both modules share,
  try {
    if (auto* pointer = object.cast<franka::ActiveControlBase*>()) {
      return pointer;
    }
  } catch (const py::cast_error&) {
  }
  // or through the conduit, which works across pybind11 versions.
  auto* pointer =
      pybind11_conduit_v1::get_type_pointer_ephemeral<franka::ActiveControlBase>(object.ptr());
  if (pointer == nullptr) {
    PyErr_Clear();
    throw std::runtime_error(
        "could not get the C++ ActiveControlBase of robot.torque_controller from pylibfranka; "
        "rebuild aiofranka's native module against the installed pylibfranka");
  }
  return pointer;
}

franka::RobotState* robot_state_pointer(py::handle object) {
  try {
    if (auto* pointer = object.cast<franka::RobotState*>()) {
      return pointer;
    }
  } catch (const py::cast_error&) {
  }
  auto* pointer = pybind11_conduit_v1::get_type_pointer_ephemeral<franka::RobotState>(object.ptr());
  if (pointer == nullptr) {
    PyErr_Clear();
    throw std::runtime_error("not a pylibfranka.RobotState");
  }
  return pointer;
}

class Loop {
 public:
  Loop() = default;
  Loop(const Loop&) = delete;
  Loop& operator=(const Loop&) = delete;

  ~Loop() {
    request_stop();
    if (thread_.joinable()) {
      if (PyGILState_Check()) {
        py::gil_scoped_release release;
        thread_.join();
      } else {
        thread_.join();
      }
    }
  }

  Params* staging() { return &staging_; }

  void set_mujoco(std::uintptr_t model, const MjArrays& arrays, int site_id, double timestep,
                  double time, const MjApi& api) {
    require_idle("set_mujoco");
    if (api.fwd_position == nullptr || api.step == nullptr || api.jac_site == nullptr ||
        api.full_m == nullptr) {
      throw std::invalid_argument("missing MuJoCo functions");
    }
    model_ = address<const void>(model);
    {
      std::lock_guard<std::mutex> lock(request_mutex_);
      pending_model_ = nullptr;  // a swap asked for before this configuration
      model_epoch_.store(requested_epoch_, std::memory_order_release);
    }
    arrays_ = arrays;
    site_id_ = site_id;
    timestep_ = timestep;
    world_time_ = time;
    mj_ = api;
  }

  // A copy of the model with a new payload, which the loop takes at its next cycle.
  int64_t swap_model(std::uintptr_t model) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    pending_model_ = address<const void>(model);
    return ++requested_epoch_;
  }

  int64_t model_epoch() const { return model_epoch_.load(std::memory_order_acquire); }

  void assign(std::size_t offset, const py::array_t<double, py::array::c_style | py::array::forcecast>& values) {
    const std::size_t bytes = static_cast<std::size_t>(values.size()) * sizeof(double);
    check_field(offset, bytes);
    std::lock_guard<std::mutex> lock(staging_mutex_);
    std::memcpy(reinterpret_cast<char*>(&staging_) + offset, values.data(), bytes);
  }

  void assign_int(std::size_t offset, int64_t value) {
    check_field(offset, sizeof(int64_t));
    std::lock_guard<std::mutex> lock(staging_mutex_);
    std::memcpy(reinterpret_cast<char*>(&staging_) + offset, &value, sizeof(value));
  }

  // The mode and custom law together, so that no cycle sees one without the other.
  void assign_mode(int64_t mode, int64_t custom_law) {
    if (mode == kCustom && (custom_law < 0 || custom_law >= law_count_.load(std::memory_order_acquire))) {
      throw std::invalid_argument("no custom law " + std::to_string(custom_law));
    }
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_.mode = mode;
    staging_.custom_law = custom_law;
  }

  void reset_integral(const py::array_t<double, py::array::c_style | py::array::forcecast>& values) {
    if (values.size() != kJoints) {
      throw std::invalid_argument("error_integral must have 7 values");
    }
    std::lock_guard<std::mutex> lock(request_mutex_);
    std::copy(values.data(), values.data() + kJoints, integral_request_);
    integral_requested_ = true;
  }

  void reset_memory() {
    std::lock_guard<std::mutex> lock(request_mutex_);
    memory_requested_ = true;
  }

  int register_law(const std::string& name, std::uintptr_t fn, int64_t memory_size) {
    if (memory_size < 0 || memory_size > kMemory) {
      throw std::invalid_argument("a law's memory holds at most " + std::to_string(kMemory) +
                                  " doubles");
    }
    const int index = law_count_.load(std::memory_order_acquire);
    if (index >= kMaxLaws) {
      throw std::runtime_error("at most " + std::to_string(kMaxLaws) + " custom laws");
    }
    laws_[index] = CustomLaw{name, reinterpret_cast<LawFn>(fn), memory_size};
    law_count_.store(index + 1, std::memory_order_release);
    return index;
  }

  void start(py::object active_control, const Realtime& rt) {
    require_idle("start");
    if (model_ == nullptr) {
      throw std::runtime_error("set_mujoco() first");
    }
    franka::ActiveControlBase* control =
        active_control.is_none() ? nullptr : active_control_pointer(active_control);
    if (thread_.joinable()) {
      thread_.join();  // a loop that ended on its own
    }
    real_ = control != nullptr;
    active_control_ = control;
    active_control_owner_ = active_control;
    rt_ = rt;
    reset_for_start();
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] { thread_main(); });
  }

  // One cycle in the calling thread, for FrankaController.step().
  void step_once(py::object active_control) {
    require_idle("step");
    if (model_ == nullptr) {
      throw std::runtime_error("set_mujoco() first");
    }
    franka::ActiveControlBase* control =
        active_control.is_none() ? nullptr : active_control_pointer(active_control);
    stepping_.store(true);
    struct Done {
      std::atomic<bool>& stepping;
      ~Done() { stepping.store(false); }
    } done{stepping_};
    py::gil_scoped_release release;
    if (control != nullptr) {
      cycle_real(control);
    } else {
      cycle_sim();
    }
  }

  void request_stop() { stop_.store(true, std::memory_order_release); }

  void join() {
    {
      py::gil_scoped_release release;
      if (thread_.joinable()) {
        thread_.join();
      }
      std::lock_guard<std::mutex> lock(recorder_mutex_);
      if (!stepping_.load()) {
        retired_recorders_.clear();
      }
    }
    active_control_owner_ = py::none();
    active_control_ = nullptr;
  }

  // Records into a recorder from the next cycle on, or stops recording with None. A recorder
  // that a cycle may still be writing to is released only once that cycle has ended.
  void set_recorder(std::shared_ptr<Recorder> recorder) {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(recorder_mutex_);
    std::shared_ptr<Recorder> old = std::move(recorder_owner_);
    recorder_owner_ = std::move(recorder);
    recorder_.store(recorder_owner_.get());
    if (old != nullptr) {
      retired_recorders_.push_back(std::move(old));
    }
    if (!retired_recorders_.empty() && no_cycle_since_swap()) {
      retired_recorders_.clear();
    }
  }

  bool running() const { return running_.load(std::memory_order_acquire); }

  std::string error() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_;
  }

  std::string realtime_status() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return realtime_status_;
  }

  // controller.state, or None before the first cycle.
  py::object state_dict() {
    LawState s{};
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      if (snapshot_.cycle == 0) {
        return py::none();
      }
      s = snapshot_.state;
    }
    return state_to_dict(s);
  }

  py::object snapshot() {
    auto snap = std::make_unique<Snapshot>();
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      if (snapshot_.cycle == 0) {
        return py::none();
      }
      copy_snapshot(snapshot_, *snap);
    }
    py::dict out;
    out["cycle"] = snap->cycle;
    out["mode"] = snap->mode;
    out["time"] = snap->state.time;
    out["state"] = state_to_dict(snap->state);
    out["torque"] = vector(snap->torque, kJoints);
    out["last_command"] = vector(snap->last_command, kJoints);
    out["error_integral"] = vector(snap->error_integral, kJoints);
    out["model_epoch"] = snap->model_epoch;
    out["memory"] = vector(snap->memory, snap->memory_size);
    return out;
  }

  // Copies the arm's latest positions, velocities and torques (in simulation, those after
  // the step) into robot.data's arrays. Returns (cycle, time, model_epoch), or None.
  py::object sync_world(const py::array& qpos, const py::array& qvel, const py::array& ctrl) {
    double* q = writable_array(qpos, "qpos");
    double* dq = writable_array(qvel, "qvel");
    double* u = writable_array(ctrl, "ctrl");
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    if (snapshot_.cycle == 0) {
      return py::none();
    }
    std::copy(snapshot_.world_qpos, snapshot_.world_qpos + kJoints, q);
    std::copy(snapshot_.world_qvel, snapshot_.world_qvel + kJoints, dq);
    std::copy(snapshot_.world_ctrl, snapshot_.world_ctrl + kJoints, u);
    return py::make_tuple(snapshot_.cycle, snapshot_.world_time, snapshot_.model_epoch);
  }

  // The last robot state as a new pylibfranka.RobotState, or None. Also None when this
  // module does not share pybind11's internals with pylibfranka, which cannot create its
  // objects then: copy_robot_state_into() updates one instead.
  py::object robot_state() {
    if (!pylibfranka_types()) {
      return py::none();
    }
    franka::RobotState copy;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      if (!has_robot_state_) {
        return py::none();
      }
      copy = robot_state_;
    }
    return py::cast(std::move(copy));
  }

  bool copy_robot_state_into(py::handle object) {
    franka::RobotState* target = robot_state_pointer(object);
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    if (!has_robot_state_) {
      return false;
    }
    *target = robot_state_;
    return true;
  }

  py::dict stats(bool reset) {
    Stats s;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      s = shared_stats_;
      if (reset) {
        shared_stats_.reset_window();
      }
    }
    py::dict out;
    out["count"] = s.count;
    out["mean"] = s.count ? s.sum / s.count : 0.0;
    out["std"] = s.count ? std::sqrt(std::max(0.0, s.sum_sq / s.count - (s.sum / s.count) * (s.sum / s.count))) : 0.0;
    out["min"] = s.min;
    out["max"] = s.max;
    out["busy_mean"] = s.busy_count ? s.busy_sum / s.busy_count : 0.0;
    out["busy_max"] = s.busy_max;
    out["robot_gap_max"] = s.robot_gap_max;
    out["missed"] = s.missed;
    out["success_min"] = s.success_min;
    out["max_all"] = s.max_all;
    out["warn"] = s.warn;
    out["error"] = s.error;
    return out;
  }

 private:
  void require_idle(const char* what) const {
    if (running()) {
      throw std::runtime_error(std::string("cannot ") + what + " while the native loop runs");
    }
  }

  // After recorder_ changed: waits until no cycle that may have read the old value still
  // runs, which is when one more cycle has ended. False if that took over 2 s. Called
  // without the GIL.
  bool no_cycle_since_swap() {
    if (!running() && !stepping_.load()) {
      return true;
    }
    const int64_t seen = cycles_done_.load();
    const Clock::time_point deadline = Clock::now() + std::chrono::seconds(2);
    while (cycles_done_.load() == seen) {
      if (!running() && !stepping_.load()) {
        return true;
      }
      if (Clock::now() > deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
  }

  static void check_field(std::size_t offset, std::size_t bytes) {
    if (offset % 8 != 0 || offset + bytes > sizeof(Params)) {
      throw std::out_of_range("no such field in Params");
    }
  }

  static py::array_t<double> vector(const double* values, int64_t n) {
    py::array_t<double> out(static_cast<py::ssize_t>(n));
    std::copy(values, values + n, out.mutable_data());
    return out;
  }

  static py::array_t<double> matrix(const double* values, int rows, int cols) {
    py::array_t<double> out({rows, cols});
    std::copy(values, values + rows * cols, out.mutable_data());
    return out;
  }

  static py::dict state_to_dict(const LawState& s) {
    py::dict out;
    out["qpos"] = vector(s.qpos, kJoints);
    out["qvel"] = vector(s.qvel, kJoints);
    out["ee"] = matrix(s.ee, 4, 4);
    out["jac"] = matrix(s.jac, 6, 7);
    out["mm"] = matrix(s.mm, 7, 7);
    out["last_torque"] = vector(s.last_torque, kJoints);
    return out;
  }

  static void copy_snapshot(const Snapshot& from, Snapshot& to) {
    // Snapshot is trivially copyable; copy only the memory a law uses.
    std::memcpy(static_cast<void*>(&to), &from, offsetof(Snapshot, memory));
    std::copy(from.memory, from.memory + from.memory_size, to.memory);
  }

  void set_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_.empty()) {
      error_ = message;
    }
  }

  void reset_for_start() {
    stop_.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_.clear();
      realtime_status_.clear();
    }
    have_last_start_ = false;
    local_stats_ = Stats{};
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    shared_stats_ = Stats{};
  }

  void thread_main() {
    const std::string status = apply_realtime(rt_);
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      realtime_status_ = status;
    }
    try {
      if (real_) {
        while (!stop_.load(std::memory_order_acquire)) {
          cycle_real(active_control_);
        }
      } else {
        const auto period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(timestep_));
        auto next = Clock::now();
        while (!stop_.load(std::memory_order_acquire)) {
          cycle_sim();
          next += period;
          const auto now = Clock::now();
          if (next < now - 20 * period) {
            next = now;  // fell far behind: do not race to catch up
          }
          std::this_thread::sleep_until(next);
        }
      }
    } catch (const std::exception& e) {
      set_error(e.what());
    } catch (...) {
      set_error("unknown error in the native control loop");
    }
    running_.store(false, std::memory_order_release);
  }

  // Takes Python's changes: the staging Params, a new model, and the resets it asked for.
  // Returns when the cycle started.
  Clock::time_point begin_cycle() {
    const Clock::time_point now = Clock::now();
    wall_start_ = std::chrono::system_clock::now();
    if (have_last_start_) {
      local_stats_.add(std::chrono::duration<double>(now - last_start_).count());
    }
    last_start_ = now;
    have_last_start_ = true;

    {
      std::unique_lock<std::mutex> lock(staging_mutex_, std::try_to_lock);
      if (lock.owns_lock()) {
        std::memcpy(&active_, &staging_, sizeof(Params));
      }
    }
    std::unique_lock<std::mutex> lock(request_mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
      if (pending_model_ != nullptr) {
        model_ = pending_model_;
        pending_model_ = nullptr;
        model_epoch_.store(requested_epoch_, std::memory_order_release);
      }
      if (integral_requested_) {
        std::copy(integral_request_, integral_request_ + kJoints, integral_);
        integral_requested_ = false;
      }
      if (memory_requested_) {
        std::fill(memory_, memory_ + kMemory, 0.0);
        memory_requested_ = false;
      }
    }
    return now;
  }

  // The flange pose, its Jacobian and the mass matrix, from the kinematics in the mjData.
  void read_kinematics(LawState& s) {
    const double* x = arrays_.site_xpos;
    const double* r = arrays_.site_xmat;
    const double ee[16] = {r[0], r[1], r[2], x[0], r[3], r[4], r[5], x[1],
                           r[6], r[7], r[8], x[2], 0.0,  0.0,  0.0,  1.0};
    std::copy(ee, ee + 16, s.ee);
    double jacp[21], jacr[21];
    mj_.jac_site(model_, arrays_.data, jacp, jacr, site_id_);
    std::copy(jacp, jacp + 21, s.jac);
    std::copy(jacr, jacr + 21, s.jac + 21);
    mj_.full_m(model_, arrays_.data, s.mm);
  }

  void compute(const LawState& s, double tau[7]) {
    const Params& p = active_;
    switch (p.mode) {
      case kImpedance:
        impedance(s, p, tau);
        std::copy(tau, tau + kJoints, torque_);
        break;
      case kPid:
        pid(s, p, integral_, tau);
        std::copy(tau, tau + kJoints, torque_);
        break;
      case kOsc:
        osc(s, p, tau);
        break;
      case kTorque:
        std::copy(p.torque, p.torque + kJoints, tau);
        break;
      case kCustom: {
        const int count = law_count_.load(std::memory_order_acquire);
        if (p.custom_law < 0 || p.custom_law >= count) {
          throw std::runtime_error("unknown custom control law " + std::to_string(p.custom_law));
        }
        const CustomLaw& law = laws_[p.custom_law];
        std::fill(tau, tau + kJoints, 0.0);
        const int32_t code = law.fn(&s, &p, memory_, tau);
        if (code != 0) {
          throw std::runtime_error("control law '" + law.name + "' returned " +
                                   std::to_string(code));
        }
        if (p.clip) {
          limit_rate(p, s.last_torque, tau);
          clip_torque(p, tau);
        }
        break;
      }
      default:
        throw std::runtime_error("Unknown controller type: " + std::to_string(p.mode));
    }
  }

  void cycle_real(franka::ActiveControlBase* control) {
    const std::pair<franka::RobotState, franka::Duration> read = control->readOnce();
    const franka::RobotState& robot = read.first;
    const Clock::time_point start = begin_cycle();

    std::copy(robot.q.begin(), robot.q.end(), arrays_.qpos);
    std::copy(robot.dq.begin(), robot.dq.end(), arrays_.qvel);
    std::copy(robot.tau_J_d.begin(), robot.tau_J_d.end(), arrays_.ctrl);
    mj_.fwd_position(model_, arrays_.data);

    LawState s{};
    s.cycle = cycle_;
    s.time = robot.time.toSec();
    s.dt = kDt;
    std::copy(robot.q.begin(), robot.q.end(), s.qpos);
    std::copy(robot.dq.begin(), robot.dq.end(), s.qvel);
    std::copy(robot.tau_J_d.begin(), robot.tau_J_d.end(), s.last_torque);
    read_kinematics(s);

    double tau[7];
    compute(s, tau);
    std::array<double, 7> command{};
    std::copy(tau, tau + kJoints, command.begin());
    control->writeOnce(franka::Torques(command));
    const double busy = std::chrono::duration<double>(Clock::now() - start).count();
    local_stats_.add_busy(busy);
    local_stats_.add_robot(read.second.toSec(), robot.control_command_success_rate);
    finish_cycle(s, tau, s.qpos, s.qvel, s.last_torque, s.time, &robot, busy);
  }

  void cycle_sim() {
    const Clock::time_point start = begin_cycle();
    // Like FrankaController in simulation: the state after the last step, with the
    // kinematics that step computed.
    LawState s{};
    s.cycle = cycle_;
    s.time = world_time_;
    s.dt = kDt;
    std::copy(arrays_.qpos, arrays_.qpos + kJoints, s.qpos);
    std::copy(arrays_.qvel, arrays_.qvel + kJoints, s.qvel);
    std::copy(arrays_.ctrl, arrays_.ctrl + kJoints, s.last_torque);
    read_kinematics(s);

    double tau[7];
    compute(s, tau);
    std::copy(tau, tau + kJoints, arrays_.ctrl);
    mj_.step(model_, arrays_.data);
    world_time_ += timestep_;
    const double busy = std::chrono::duration<double>(Clock::now() - start).count();
    local_stats_.add_busy(busy);
    finish_cycle(s, tau, arrays_.qpos, arrays_.qvel, arrays_.ctrl, world_time_, nullptr, busy);
  }

  // A row of the recording, if there is one.
  void record(const LawState& s, const double tau[7], const franka::RobotState* robot,
              double busy) {
    Recorder* recorder = recorder_.load();
    if (recorder != nullptr) {
      RecordExtra extra;
      extra.wall_time = std::chrono::duration<double>(wall_start_.time_since_epoch()).count();
      extra.busy = busy;
      extra.robot_mode = robot != nullptr ? static_cast<int64_t>(robot->robot_mode) : -1;
      std::copy(tau, tau + kJoints, extra.tau);
      la::Mat<4, 4> flange{}, transform{};
      std::copy(s.ee, s.ee + 16, flange.begin());
      std::copy(active_.control_transform, active_.control_transform + 16, transform.begin());
      const la::Mat<4, 4> tcp = la::matmul<4, 4, 4>(flange, transform);
      std::copy(tcp.begin(), tcp.end(), extra.tcp);
      const void* const sources[kSources] = {&s, &active_, robot, &extra};
      recorder->push(sources);
    }
    cycles_done_.fetch_add(1);
  }

  void finish_cycle(const LawState& s, const double tau[7], const double* world_qpos,
                    const double* world_qvel, const double* world_ctrl, double world_time,
                    const franka::RobotState* robot, double busy) {
    record(s, tau, robot, busy);
    ++cycle_;
    std::unique_lock<std::mutex> lock(snapshot_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return;  // Python is copying the snapshot; the next cycle publishes
    }
    Snapshot& snap = snapshot_;
    snap.cycle = cycle_;
    snap.mode = active_.mode;
    snap.state = s;
    std::copy(torque_, torque_ + kJoints, snap.torque);
    std::copy(tau, tau + kJoints, snap.last_command);
    std::copy(integral_, integral_ + kJoints, snap.error_integral);
    std::copy(world_qpos, world_qpos + kJoints, snap.world_qpos);
    std::copy(world_qvel, world_qvel + kJoints, snap.world_qvel);
    std::copy(world_ctrl, world_ctrl + kJoints, snap.world_ctrl);
    snap.world_time = world_time;
    snap.model_epoch = model_epoch_.load(std::memory_order_relaxed);
    int64_t memory_size = 0;
    if (active_.mode == kCustom && active_.custom_law >= 0 &&
        active_.custom_law < law_count_.load(std::memory_order_acquire)) {
      memory_size = laws_[active_.custom_law].memory_size;
    }
    snap.memory_size = memory_size;
    std::copy(memory_, memory_ + memory_size, snap.memory);
    if (robot != nullptr) {
      robot_state_ = *robot;
      has_robot_state_ = true;
    }
    shared_stats_.merge(local_stats_);
    local_stats_ = Stats{};
  }

  // MuJoCo
  MjApi mj_{};
  const void* model_ = nullptr;
  MjArrays arrays_{};
  int site_id_ = 0;
  double timestep_ = 1e-3;
  double world_time_ = 0;

  // robot
  bool real_ = false;
  franka::ActiveControlBase* active_control_ = nullptr;
  py::object active_control_owner_ = py::none();  // keeps pylibfranka's object alive
  Realtime rt_{};

  // controller attributes
  std::mutex staging_mutex_;
  Params staging_{};
  Params active_{};

  // requests from Python
  std::mutex request_mutex_;
  const void* pending_model_ = nullptr;
  int64_t requested_epoch_ = 0;
  std::atomic<int64_t> model_epoch_{0};
  bool integral_requested_ = false;
  double integral_request_[7]{};
  bool memory_requested_ = false;

  // owned by the loop
  int64_t cycle_ = 0;
  double integral_[7]{};
  double torque_[7]{};
  double memory_[kMemory]{};
  Clock::time_point last_start_{};
  std::chrono::system_clock::time_point wall_start_{};
  bool have_last_start_ = false;
  Stats local_stats_{};

  // recording: recorder_ is what cycles write to; the owners keep it alive
  std::atomic<Recorder*> recorder_{nullptr};
  std::mutex recorder_mutex_;  // Python's, for the owners
  std::shared_ptr<Recorder> recorder_owner_;
  std::vector<std::shared_ptr<Recorder>> retired_recorders_;  // a stalled cycle may still use
  std::atomic<int64_t> cycles_done_{0};
  std::atomic<bool> stepping_{false};

  // custom laws
  std::array<CustomLaw, kMaxLaws> laws_{};
  std::atomic<int> law_count_{0};

  // published for Python
  std::mutex snapshot_mutex_;
  Snapshot snapshot_{};
  franka::RobotState robot_state_{};
  bool has_robot_state_ = false;
  Stats shared_stats_{};

  // thread
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};
  std::mutex error_mutex_;
  std::string error_;
  std::string realtime_status_;
};

MjApi mj_api(const py::dict& functions) {
  MjApi api;
  api.fwd_position = reinterpret_cast<decltype(api.fwd_position)>(
      functions["mj_fwdPosition"].cast<std::uintptr_t>());
  api.step = reinterpret_cast<decltype(api.step)>(functions["mj_step"].cast<std::uintptr_t>());
  api.jac_site =
      reinterpret_cast<decltype(api.jac_site)>(functions["mj_jacSite"].cast<std::uintptr_t>());
  api.full_m = reinterpret_cast<decltype(api.full_m)>(functions["mj_fullM"].cast<std::uintptr_t>());
  return api;
}

MjArrays mj_arrays(const py::dict& a) {
  MjArrays arrays;
  arrays.data = address<void>(a["data"].cast<std::uintptr_t>());
  arrays.qpos = address<double>(a["qpos"].cast<std::uintptr_t>());
  arrays.qvel = address<double>(a["qvel"].cast<std::uintptr_t>());
  arrays.ctrl = address<double>(a["ctrl"].cast<std::uintptr_t>());
  arrays.site_xpos = address<const double>(a["site_xpos"].cast<std::uintptr_t>());
  arrays.site_xmat = address<const double>(a["site_xmat"].cast<std::uintptr_t>());
  return arrays;
}

py::tuple field(const char* name, std::size_t offset, py::tuple shape, const char* dtype) {
  return py::make_tuple(name, offset, shape, dtype);
}

// The numbers in franka::RobotState, by offset, for recordings. Matrices are flat and
// column-major, as in pylibfranka.
py::tuple robot_layout() {
  const franka::RobotState r{};
  const char* base = reinterpret_cast<const char*>(&r);
  py::list fields;
  auto add = [&](const char* name, const double* values, py::tuple shape) {
    const auto offset = static_cast<std::size_t>(reinterpret_cast<const char*>(values) - base);
    fields.append(field(name, offset, std::move(shape), "f8"));
  };
#define AIOFRANKA_ARRAY(name) add(#name, r.name.data(), py::make_tuple(r.name.size()))
#define AIOFRANKA_SCALAR(name) add(#name, &r.name, py::make_tuple())
  AIOFRANKA_ARRAY(O_T_EE);
  AIOFRANKA_ARRAY(O_T_EE_d);
  AIOFRANKA_ARRAY(F_T_EE);
  AIOFRANKA_ARRAY(F_T_NE);
  AIOFRANKA_ARRAY(NE_T_EE);
  AIOFRANKA_ARRAY(EE_T_K);
  AIOFRANKA_SCALAR(m_ee);
  AIOFRANKA_ARRAY(I_ee);
  AIOFRANKA_ARRAY(F_x_Cee);
  AIOFRANKA_SCALAR(m_load);
  AIOFRANKA_ARRAY(I_load);
  AIOFRANKA_ARRAY(F_x_Cload);
  AIOFRANKA_SCALAR(m_total);
  AIOFRANKA_ARRAY(I_total);
  AIOFRANKA_ARRAY(F_x_Ctotal);
  AIOFRANKA_ARRAY(elbow);
  AIOFRANKA_ARRAY(elbow_d);
  AIOFRANKA_ARRAY(elbow_c);
  AIOFRANKA_ARRAY(delbow_c);
  AIOFRANKA_ARRAY(ddelbow_c);
  AIOFRANKA_ARRAY(tau_J);
  AIOFRANKA_ARRAY(tau_J_d);
  AIOFRANKA_ARRAY(dtau_J);
  AIOFRANKA_ARRAY(q);
  AIOFRANKA_ARRAY(q_d);
  AIOFRANKA_ARRAY(dq);
  AIOFRANKA_ARRAY(dq_d);
  AIOFRANKA_ARRAY(ddq_d);
  AIOFRANKA_ARRAY(joint_contact);
  AIOFRANKA_ARRAY(cartesian_contact);
  AIOFRANKA_ARRAY(joint_collision);
  AIOFRANKA_ARRAY(cartesian_collision);
  AIOFRANKA_ARRAY(tau_ext_hat_filtered);
  AIOFRANKA_ARRAY(O_F_ext_hat_K);
  AIOFRANKA_ARRAY(K_F_ext_hat_K);
  AIOFRANKA_ARRAY(O_dP_EE_d);
  AIOFRANKA_ARRAY(O_ddP_O);
  AIOFRANKA_ARRAY(O_T_EE_c);
  AIOFRANKA_ARRAY(O_dP_EE_c);
  AIOFRANKA_ARRAY(O_ddP_EE_c);
  AIOFRANKA_ARRAY(theta);
  AIOFRANKA_ARRAY(dtheta);
  AIOFRANKA_SCALAR(control_command_success_rate);
#undef AIOFRANKA_ARRAY
#undef AIOFRANKA_SCALAR
  return py::tuple(fields);
}

}  // namespace
}  // namespace aiofranka

PYBIND11_MODULE(_native, m) {
  using namespace aiofranka;
  m.doc() = "The C++ control loop of aiofranka.NativeFrankaController.";

  m.attr("LIBFRANKA_VERSION") = kLibfrankaVersion;
  m.attr("CUSTOM_PARAMS") = kCustomParams;
  m.attr("MEMORY") = kMemory;
  m.attr("MODES") = py::dict(py::arg("impedance") = int(kImpedance), py::arg("pid") = int(kPid),
                             py::arg("osc") = int(kOsc), py::arg("torque") = int(kTorque),
                             py::arg("custom") = int(kCustom));

  const py::tuple scalar = py::make_tuple();
  const py::tuple seven = py::make_tuple(7);
  const py::tuple six = py::make_tuple(6);
  const py::tuple pose = py::make_tuple(4, 4);
  m.attr("PARAMS_SIZE") = sizeof(Params);
  m.attr("PARAMS_LAYOUT") = py::make_tuple(
      field("mode", offsetof(Params, mode), scalar, "i8"),
      field("clip", offsetof(Params, clip), scalar, "i8"),
      field("custom_law", offsetof(Params, custom_law), scalar, "i8"),
      field("torque_diff_limit", offsetof(Params, torque_diff_limit), seven, "f8"),
      field("kp", offsetof(Params, kp), seven, "f8"),
      field("kd", offsetof(Params, kd), seven, "f8"),
      field("ki", offsetof(Params, ki), seven, "f8"),
      field("ee_kp", offsetof(Params, ee_kp), six, "f8"),
      field("ee_kd", offsetof(Params, ee_kd), six, "f8"),
      field("null_kp", offsetof(Params, null_kp), seven, "f8"),
      field("null_kd", offsetof(Params, null_kd), seven, "f8"),
      field("q_desired", offsetof(Params, q_desired), seven, "f8"),
      field("ee_desired", offsetof(Params, ee_desired), pose, "f8"),
      field("torque", offsetof(Params, torque), seven, "f8"),
      field("initial_qpos", offsetof(Params, initial_qpos), seven, "f8"),
      field("control_transform", offsetof(Params, control_transform), pose, "f8"),
      field("torque_limit", offsetof(Params, torque_limit), seven, "f8"),
      field("custom", offsetof(Params, custom), py::make_tuple(kCustomParams), "f8"));
  m.attr("STATE_SIZE") = sizeof(LawState);
  m.attr("STATE_LAYOUT") = py::make_tuple(
      field("cycle", offsetof(LawState, cycle), scalar, "i8"),
      field("time", offsetof(LawState, time), scalar, "f8"),
      field("dt", offsetof(LawState, dt), scalar, "f8"),
      field("qpos", offsetof(LawState, qpos), seven, "f8"),
      field("qvel", offsetof(LawState, qvel), seven, "f8"),
      field("ee", offsetof(LawState, ee), pose, "f8"),
      field("jac", offsetof(LawState, jac), py::make_tuple(6, 7), "f8"),
      field("mm", offsetof(LawState, mm), py::make_tuple(7, 7), "f8"),
      field("last_torque", offsetof(LawState, last_torque), seven, "f8"));
  m.attr("ROBOT_LAYOUT") = robot_layout();
  m.attr("RECORD_EXTRA_LAYOUT") = py::make_tuple(
      field("wall_time", offsetof(RecordExtra, wall_time), scalar, "f8"),
      field("busy", offsetof(RecordExtra, busy), scalar, "f8"),
      field("robot_mode", offsetof(RecordExtra, robot_mode), scalar, "i8"),
      field("tau", offsetof(RecordExtra, tau), seven, "f8"),
      field("tcp", offsetof(RecordExtra, tcp), pose, "f8"));
  m.attr("RECORD_SOURCES") =
      py::dict(py::arg("state") = int(kFromState), py::arg("params") = int(kFromParams),
               py::arg("robot") = int(kFromRobot), py::arg("extra") = int(kFromExtra));

  m.def("has_pylibfranka_types", &pylibfranka_types);

  py::class_<Recorder, std::shared_ptr<Recorder>>(
      m, "Recorder", "Chosen fields of every cycle, in a ring that the loop fills and drain() empties.")
      .def(py::init([](const std::vector<std::tuple<int, std::size_t, std::size_t>>& slices,
                       int64_t capacity) {
             std::vector<Recorder::Slice> parts;
             for (const auto& [source, offset, bytes] : slices) {
               parts.push_back(Recorder::Slice{source, offset, bytes});
             }
             return std::make_shared<Recorder>(std::move(parts), capacity);
           }),
           py::arg("slices"), py::arg("capacity"))
      .def("drain", &Recorder::drain)
      .def_property_readonly("row_bytes", &Recorder::row_bytes)
      .def_property_readonly("capacity", &Recorder::capacity)
      .def_property_readonly("rows", &Recorder::rows)
      .def_property_readonly("dropped", &Recorder::dropped);

  py::class_<FakeActiveControl>(m, "_FakeActiveControl",
                                "A simulated robot for tests, which answers readOnce() and "
                                "writeOnce() at 1 kHz like the real one.")
      .def(py::init([](std::uintptr_t model, const py::dict& world, std::uintptr_t step,
                       double period, int64_t fail_after, std::string fail_message) {
             return new FakeActiveControl(model, mj_arrays(world), step, period, fail_after,
                                          std::move(fail_message));
           }),
           py::arg("model"), py::arg("world"), py::arg("step"), py::arg("period") = 1e-3,
           py::arg("fail_after") = -1, py::arg("fail_message") = "")
      .def_property_readonly("reads", &FakeActiveControl::reads)
      .def_property_readonly("writes", &FakeActiveControl::writes)
      .def_property_readonly("max_gap", &FakeActiveControl::max_gap)
      .def("reset_max_gap", &FakeActiveControl::reset_max_gap)
      // pylibfranka's ActiveControlBase methods, so FrankaController can drive it too.
      .def("readOnce",
           [](FakeActiveControl& self) {
             if (!pylibfranka_types()) {
               throw std::runtime_error("readOnce() from Python needs pylibfranka's pybind11 internals");
             }
             std::pair<franka::RobotState, franka::Duration> read;
             {
               py::gil_scoped_release release;
               read = self.readOnce();
             }
             return py::make_tuple(py::cast(read.first), py::cast(read.second));
           })
      .def("writeOnce",
           [](FakeActiveControl& self, const franka::Torques& torques) { self.writeOnce(torques); });

  py::class_<Loop>(m, "Loop")
      .def(py::init<>())
      .def("params_buffer",
           [](py::object self) {
             Loop& loop = self.cast<Loop&>();
             return py::array(py::dtype("uint8"), {static_cast<py::ssize_t>(sizeof(Params))},
                              {static_cast<py::ssize_t>(1)},
                              reinterpret_cast<uint8_t*>(loop.staging()), self);
           })
      .def("set_mujoco",
           [](Loop& loop, std::uintptr_t model, const py::dict& arrays, int site_id,
              double timestep, double time, const py::dict& functions) {
             loop.set_mujoco(model, mj_arrays(arrays), site_id, timestep, time, mj_api(functions));
           },
           py::arg("model"), py::arg("arrays"), py::arg("site_id"), py::arg("timestep"),
           py::arg("time"), py::arg("functions"))
      .def("swap_model", &Loop::swap_model)
      .def("model_epoch", &Loop::model_epoch)
      .def("assign", &Loop::assign)
      .def("assign_int", &Loop::assign_int)
      .def("assign_mode", &Loop::assign_mode)
      .def("reset_integral", &Loop::reset_integral)
      .def("reset_memory", &Loop::reset_memory)
      .def("register_law", &Loop::register_law)
      .def("start",
           [](Loop& loop, py::object active_control, bool macos_qos, int cpu, int fifo_priority) {
             loop.start(std::move(active_control), Realtime{macos_qos, cpu, fifo_priority});
           },
           py::arg("active_control"), py::arg("macos_qos") = true, py::arg("cpu") = -1,
           py::arg("fifo_priority") = 0)
      .def("step_once", &Loop::step_once)
      .def("request_stop", &Loop::request_stop)
      .def("join", &Loop::join)
      .def("running", &Loop::running)
      .def("error", &Loop::error)
      .def("realtime_status", &Loop::realtime_status)
      .def("state_dict", &Loop::state_dict)
      .def("snapshot", &Loop::snapshot)
      .def("sync_world", &Loop::sync_world)
      .def("robot_state", &Loop::robot_state)
      .def("copy_robot_state_into", &Loop::copy_robot_state_into)
      .def("set_recorder", &Loop::set_recorder, py::arg("recorder").none(true))
      .def("stats", &Loop::stats, py::arg("reset") = false);
}
