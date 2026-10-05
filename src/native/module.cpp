// aiofranka._native: the 1 kHz control loop of NativeFrankaController, in C++.
//
// The loop runs in its own thread and never touches Python. Each cycle it reads the robot
// state through the ActiveControl that pylibfranka started, computes the flange pose,
// Jacobian and mass matrix with the libmujoco that the mujoco package loaded, runs the
// control law, and sends the torques. Python writes the controller's attributes into a
// staging copy of Params, which the loop copies at the start of each cycle, and reads what
// the loop did from a snapshot. Neither side waits for the other: the loop only try-locks.
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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

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
    py::gil_scoped_release release;
    if (control != nullptr) {
      cycle_real(control);
    } else {
      cycle_sim();
    }
  }

  void request_stop() { stop_.store(true, std::memory_order_release); }

  void join() {
    if (thread_.joinable()) {
      py::gil_scoped_release release;
      thread_.join();
    }
    active_control_owner_ = py::none();
    active_control_ = nullptr;
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
    local_stats_.add_busy(std::chrono::duration<double>(Clock::now() - start).count());
    local_stats_.add_robot(read.second.toSec(), robot.control_command_success_rate);
    finish_cycle(s, tau, s.qpos, s.qvel, s.last_torque, s.time, &robot);
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
    local_stats_.add_busy(std::chrono::duration<double>(Clock::now() - start).count());
    finish_cycle(s, tau, arrays_.qpos, arrays_.qvel, arrays_.ctrl, world_time_, nullptr);
  }

  void finish_cycle(const LawState& s, const double tau[7], const double* world_qpos,
                    const double* world_qvel, const double* world_ctrl, double world_time,
                    const franka::RobotState* robot) {
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
  bool have_last_start_ = false;
  Stats local_stats_{};

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

  m.def("has_pylibfranka_types", &pylibfranka_types);

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
      .def("stats", &Loop::stats, py::arg("reset") = false);
}
