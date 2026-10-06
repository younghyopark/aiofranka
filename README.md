# aiofranka

<div align="center">
  <img width="340" src="assets/image.png">
</div>
<p align="center">
  <a href="https://pypi.org/project/aiofranka/">
    <img src="https://img.shields.io/pypi/v/aiofranka" alt="CI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="CI">
  </a>
</p>

> **Note:** This repository was transferred from `Improbable-AI/aiofranka` to `younghyopark/aiofranka`. Old links and git remotes redirect here automatically.

**aiofranka** is an asyncio-based Python library for controlling Franka Emika robots. It provides a high-level, asynchronous interface that combines **`pylibfranka`** for official low-level control interface (1kHz torque control), **`MuJoCo`** for kinematics/dynamics computation, **`Ruckig`** for  smooth trajectory generation.

Its 1 kHz control loop runs in C++, in a thread that never waits for Python. The library is designed for research applications requiring precise, real-time control with minimal latency and maximum flexibility.

## Installation

Make sure you can access Franka Desk GUI from your machine's browser by typing in the robot's IP (e.g. 172.16.0.2). Then, install:

```bash
pip install aiofranka
```

This works on:

| Platform | Python |
|----------|--------|
| Linux x86_64, e.g. Ubuntu 22.04 or newer | 3.10 to 3.12 |
| Apple Silicon Mac with macOS 15 or newer | 3.10 to 3.14 |

To calibrate a camera against the robot (`aiofranka camera`), add the camera extra, which installs [aiocamera](https://github.com/younghyopark/aiocamera) and [aprilcube](https://github.com/younghyopark/aprilcube):

```bash
pip install "aiofranka[camera]"
```

Or for development:
```bash
git clone https://github.com/younghyopark/aiofranka.git
cd aiofranka
pip install -e .
```

### macOS (Apple Silicon)

On macOS, aiofranka installs [pylibfranka-macos](https://pypi.org/project/pylibfranka-macos/), an unofficial build of pylibfranka with macOS support from [younghyopark/libfranka](https://github.com/younghyopark/libfranka/tree/macos-support). It is not affiliated with Franka Robotics. To keep up with the 1 kHz control loop, it keeps one performance core busy while a control loop runs, so plug in the Mac when controlling the robot, and connect the robot via wired Ethernet.

Other Macs need pylibfranka built from source, see the [libfranka macOS instructions](https://github.com/younghyopark/libfranka/tree/macos-support/pylibfranka#installing-prerequisites-on-macos). On an Apple Silicon Mac with macOS 14 or older, pylibfranka-macos does not install, so install aiofranka with `pip install --no-deps aiofranka` and its other dependencies yourself.

## Quick Start

There are two ways to use aiofranka. Both run the 1 kHz control loop in C++, in a thread that never waits for Python, so nothing in your script can delay a torque command (see [Native Control Loop](#native-control-loop)).

### Option A: Async mode

Run the control loop in your process with `NativeFrankaController`, from an asyncio script.

- **Single script**: no separate server process
- **Direct access**: no IPC, and the controller's attributes are the loop's memory
- **Everything native**: custom control laws compiled into the loop, and a log of every cycle with `record()`

```python
import asyncio
import numpy as np
from aiofranka import RobotInterface, NativeFrankaController

async def main():
    robot = RobotInterface("172.16.0.2")
    controller = NativeFrankaController(robot)

    await controller.start()
    await controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

    controller.switch("impedance")
    controller.kp = np.ones(7) * 80.0
    controller.kd = np.ones(7) * 4.0
    controller.set_freq(50)

    for cnt in range(100):
        delta = np.sin(cnt / 50.0 * np.pi) * 0.1
        init = controller.initial_qpos
        await controller.set("q_desired", delta + init)

    await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

Pass `None` instead of the IP to run the same script on a simulated robot in MuJoCo. `examples/` has more.

### Option B: Server mode

Run the control loop in a subprocess with `FrankaRemoteController`. Your scripts use a plain sync API, no `async`/`await` needed.

- **No `async`/`await`**: plain Python scripts, easy to integrate with existing codebases
- **Process-isolated**: the loop and your script do not share a process
- **Automatic lifecycle**: the server subprocess starts with your script and stops when it exits

```python
import numpy as np
import aiofranka
from aiofranka import FrankaRemoteController

# 1. Unlock the robot (opens brakes + activates FCI)
aiofranka.unlock()

# 2. Create controller and start server subprocess
controller = FrankaRemoteController()
controller.start()

# 3. Use the robot
controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

controller.switch("impedance")
controller.kp = np.ones(7) * 80.0
controller.kd = np.ones(7) * 4.0
controller.set_freq(50)

for cnt in range(100):
    state = controller.state
    delta = np.sin(cnt / 50.0 * np.pi) * 0.1
    controller.set("q_desired", delta + controller.initial_qpos)

# 4. Stop server and lock robot
controller.stop()
aiofranka.lock()
```

The server subprocess terminates automatically when your script exits (Ctrl+C, crash, etc.), so it won't leave orphaned processes. `controller.start()` checks that the robot is unlocked and FCI is active before launching — if not, it prints a status summary and exits cleanly.

### Legacy: the Python loop

Earlier versions ran the 1 kHz loop in Python: `FrankaController` on the asyncio event loop, the server with `aiofranka start-server --python` (or `aiofranka.start(native=False)`, `FrankaRemoteController(native=False)`), and `FrankaRemoteControllerV2` in a Python thread. They keep working, with the same API, but are soft-deprecated: anything else that runs in Python can delay their torque commands, so any blocking call over about 1 ms can stop the robot with `communication_constraints_violation` (see the [Async Mode Guide](docs/ASYNC_MODE.md)). New code should use the native loop. To port, swap `FrankaController` for `NativeFrankaController`; a subclass that overrides `step()` becomes a [control law](#custom-control-laws).

## CLI Reference

The CLI handles robot setup, server lifecycle, and diagnostics.

```
aiofranka start-server [--ip IP] [--no-home]  Start the control server
aiofranka unlock   [--ip IP]              Unlock joints + activate FCI
aiofranka lock     [--ip IP]              Lock joints + deactivate FCI
aiofranka gravcomp [--ip IP] [--mode program]  Move the robot by hand (freedrive)
aiofranka home     [--ip IP]              Move the robot to its home pose
aiofranka status   [--ip IP]              Show robot & server status
aiofranka stop     [--ip IP]              Stop a running server
aiofranka mode     [--ip IP] [program|execute]  View/change operating mode
aiofranka config   [--ip IP] [--mass M]   View/set the active end-effector profile
aiofranka tool     identify|load|unload|list|remove   Identify and switch tools
aiofranka camera   calibrate|fit          Locate a fixed camera relative to the robot
aiofranka selftest [--ip IP] [--force]    Run safety self-tests
aiofranka log      [-n LINES] [-f]        View server logs
aiofranka gripper  --open|--close          Control the Robotiq gripper
aiofranka rt-benchmark [--duration SEC]    Benchmark the 1 kHz control loop
```

### `unlock` / `lock`

Unlock opens the brakes and activates FCI so the robot is ready for torque control. It first recovers safety errors, runs the self-tests if they are overdue, and switches from Programming back to Execution. Lock does the reverse. Credentials are prompted on first use and saved to `~/.aiofranka/config.json`.

```bash
# Unlock before running your script
aiofranka unlock

# Lock when you're done
aiofranka lock
```

You can also do this from Python:

```python
import aiofranka
aiofranka.unlock()   # opens brakes + activates FCI
# ... run your control script ...
aiofranka.lock()     # closes brakes + deactivates FCI
```

### `gravcomp`

Runs gravity compensation mode in the foreground. The robot is freely movable by hand. Press Ctrl+C to stop control; the joints remain unlocked with FCI active. Run `aiofranka lock` when finished.

With `--mode program`, it switches the robot to Programming mode instead, as `aiofranka mode program` does: the arm moves by hand only while the guiding button on the end effector is held, and nothing keeps running. `aiofranka unlock` switches back for FCI.

```bash
aiofranka gravcomp                  # default: zero damping
aiofranka gravcomp --damping 2.0    # add velocity damping
aiofranka gravcomp --mode program   # hand-guide with the guiding button instead
```

### `status`

Shows robot state (joints locked/unlocked, FCI active/inactive, control token, self-test status, the active end-effector profile) and server status if running.

```bash
aiofranka status
```

### `start-server`

Runs the control server in the background for clients such as `FrankaRemoteController` (`--foreground` keeps it in the terminal). It unlocks the robot (`--no-unlock` skips that), moves it home (`--no-home` skips that) and runs the 1 kHz loop in C++ (`NativeServerController`). `--python` runs the legacy Python loop instead.

```bash
aiofranka start-server              # the native loop
aiofranka start-server --python     # the legacy Python loop
```

### `stop`

Sends a shutdown signal to a running server process. The server deactivates FCI, locks joints, and releases the control token.

```bash
aiofranka stop
```

### `mode`

View or change the operating mode. `Execution` is needed for FCI control. `Programming` enables freedrive via the pilot interface button near the end-effector, as Desk's mode switch does: `aiofranka mode program` deactivates FCI, opens the brakes if they are closed, and hands the control token back to Desk. `aiofranka unlock` switches back to Execution by itself.

```bash
aiofranka mode            # view current mode
aiofranka mode program    # switch to Programming, to hand-guide the robot
aiofranka mode execute    # switch back to Execution, for FCI
```

### `config`

View or set the end-effector configuration (mass, center of mass, inertia, flange-to-EE transform). Changes are applied via the Franka Desk API to the active end-effector profile; to keep several tools by name, use `aiofranka tool`.

```bash
aiofranka config                                # view current config
aiofranka config --mass 0.5 --com 0,0,0.03      # set mass + CoM
aiofranka config --translation 0,0,0.1           # set flange-to-EE offset
```

### `tool`

Desk keeps named end-effector profiles (Settings > End Effector): the mass, center of mass and inertia of the tool on the flange. The robot compensates the gravity of the active profile, and aiofranka merges it into the MuJoCo model when it connects, so the OSC's mass matrix includes the tool too.

```bash
aiofranka tool identify gripper   # identify the mounted tool, save it as profile "gripper", activate it
aiofranka tool load gripper       # activate a profile when its tool is mounted
aiofranka tool unload             # activate the built-in "No End Effector" profile
aiofranka tool list               # list the profiles, marking the active one
aiofranka tool remove gripper     # delete a profile
```

`tool identify` moves the robot through 16 poses around the current one (about 3 minutes), approaching each from both sides to cancel joint stiction, and fits the mass and center of mass to the joint torques at rest. Start in an open pose and keep a hand on the e-stop. The poses are checked for collisions of the arm, a cylinder around the tool (`--tool-length`, `--tool-radius`, default 0.2 m by 0.1 m) and the floor (`--floor`, default the mounting plane). It shows the estimate and asks before saving it, offering to edit the mass and center of mass, e.g. to enter a scale reading.

The inertia and the TCP cannot be identified this way; set them in the Desk web UI if needed. Tools lighter than about 200 g are better weighed: the center of mass then comes out only to about a centimeter.

From Python, with a `NativeFrankaController`:

```python
estimate = await controller.identify_payload(tool_length=0.2)   # moves the robot
aiofranka.save_tool("gripper", estimate.mass, estimate.com)
aiofranka.load_tool("gripper")
```

### `camera`

Locates a fixed camera in the robot's base frame, for example to track objects in the robot's coordinates. The arm holds an AprilCube on its flange: print aprilcube's calibration cube ([cube.3mf](https://github.com/younghyopark/aprilcube/blob/main/models/calibration_cube/cube.3mf), 1x3x3 with 24 mm tags) and mount it with its connector. Start the camera with `aiocamera start`, set the cube's mass as the active Desk profile (`aiofranka tool identify`), then:

```bash
aiofranka camera calibrate              # move the arm by hand; captures and fits into camera_calibration/<date>/
aiofranka camera fit camera_calibration/20261003_150000   # fit a recorded session again
```

`camera calibrate` switches the robot to Programming mode, as `aiofranka mode program` does, and shows a live view of the camera image in the terminal: the cube in view, the views captured so far, and which image regions still have none. Move the arm holding the guiding button on the end effector, and let go: whenever it has been still for 0.7 s at a new pose, at least 5 cm or 10 deg from every captured one, with the cube in view, it records the cube's tag corners in a fresh frame with the flange pose and beeps. FCI does not run in Programming mode, so the joint positions come from Desk, as its web UI gets them. Space captures anyway, `u` removes the last view, `q` quits keeping the views. Aim for 15 to 25 views spread over the image, near and far, with the wrist turned 20 to 40 deg about at least two axes. Enter fits. The robot stays in Programming mode; `aiofranka unlock` switches back.

The fit finds the camera's pose in the base frame and the cube's on the flange that best reproject the cube's corners in every view, with the stream's factory intrinsics fixed. Every fifth view is held out of a first fit to report the error on views it has not seen. The session folder keeps `views.json` and the images, so `camera fit` can refit it, and `calibration.json` holds `T_base_camera` (meters; camera axes right, down, forward), `T_ee_cube`, the intrinsics and the errors. The camera is the only RealSense color stream in aiocamera unless `--stream` names one; `--cube` takes another AprilCube's `config.json`.

### `selftest`

Run the robot's safety self-tests. The robot will lock joints during the test.

```bash
aiofranka selftest          # run if due
aiofranka selftest --force  # run even if not due
```

### `log`

View recent server log entries from `~/.aiofranka/server.log`.

```bash
aiofranka log              # last 20 lines
aiofranka log -n 100       # last 100 lines
aiofranka log -f           # follow (like tail -f)
```

### `rt-benchmark`

Measures the 1 kHz loop on this machine. It holds the current pose in gravcomp, impedance and OSC in turn, 10 s each (`--duration`), records every cycle and compares the modes. `--mode` picks the modes; one mode prints its full report: the periods and their percentiles, the response time from `readOnce` to `writeOnce` against a 300 us budget, skipped robot states, dropped commands and a histogram. `--python` benchmarks the legacy Python loop instead, with a breakdown per phase, and `--all-combos` compares its real-time settings.

```bash
aiofranka rt-benchmark                 # the native loop, every mode
aiofranka rt-benchmark --mode osc      # one mode, its full report
aiofranka rt-benchmark --python        # the legacy Python loop
```

### Common flags

Most commands accept these flags:

| Flag | Description |
|------|-------------|
| `--ip IP` | Robot IP address (default: last used, or `172.16.0.2`) |
| `--username USER` | Franka Desk web UI username (default: saved or prompted) |
| `--password PASS` | Franka Desk web UI password (default: saved or prompted) |
| `--protocol http\|https` | Web UI protocol (default: `https`) |

## Core Concepts

### Async Mode vs Server Mode

|                          | Async mode                          | Server mode                        |
|--------------------------|-------------------------------------|------------------------------------|
| **Class**                | `NativeFrankaController`            | `FrankaRemoteController`           |
| **API style**            | `async`/`await`                     | Synchronous (plain Python)         |
| **1kHz loop runs in**    | A C++ thread in your process        | A C++ thread in a subprocess (auto-managed) |
| **Blocking calls OK?**   | Yes for the robot; they delay your next target | Yes                     |
| **State reads**          | Direct attribute access             | Shared memory (zero-copy)          |
| **Commands**             | Direct method calls                 | ZMQ IPC (msgpack)                  |
| **Setup**                | Single script                       | `unlock()` + `ctrl.start()`        |
| **Only here**            | `set_tcp()`, `activate()`, `identify_payload()`, control laws, `record()` | —  |
| **Best for**             | Most scripts, policies, custom control laws | Code without asyncio       |

### Rate Limiting

Use `set_freq()` to enforce strict timing for command updates:

```python
controller.set_freq(50)  # Set 50Hz update rate

# This will automatically sleep to maintain 50Hz timing
for i in range(100):
    controller.set("q_desired", compute_target())
```


### State Access

Robot state is continuously updated at 1kHz and accessible via `controller.state`:

```python
state = controller.state  # Thread-safe access
# Contains: qpos, qvel, ee, jac, mm, last_torque
print(f"Joint positions: {state['qpos']}")
print(f"End-effector pose: {state['ee']}")  # 4x4 homogeneous transform
```

## Controllers

### 1. Impedance Control (Joint Space)

Controls joint positions with spring-damper behavior:

```python
controller.switch("impedance")
controller.kp = np.ones(7) * 80.0   # Position gains
controller.kd = np.ones(7) * 4.0    # Damping gains

controller.set("q_desired", target_joint_angles)
```

**Use case**: Precise joint-space motions, compliant behavior


### 2. Operational Space Control (Task Space)

Controls end-effector pose in Cartesian space:

```python
controller.switch("osc")
controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])  # [xyz, rpy]
controller.ee_kd = np.ones(6) * 10.0

desired_ee = np.eye(4)  # 4x4 homogeneous transform
desired_ee[:3, 3] = [0.4, 0.0, 0.5]  # Position
controller.set("ee_desired", desired_ee)
```

By default, the OSC controls the flange. To control a point on the tool instead, set the tool center point (TCP) as a translation or a 4x4 pose in the flange frame (async mode):

```python
controller.switch("osc")
controller.set_tcp([0, 0, 0.1034])  # e.g. the Franka Hand fingertips; the arm holds still
```

The flange frame has its origin at the center of the flange face and z pointing out of it (x red, y green, z blue; right: a TCP 10 cm along z):

![The flange frame of the FR3](docs/source/images/flange_frame.png)

**Use case**: Cartesian trajectories, end-effector tracking

## Native Control Loop

The legacy `FrankaController` runs its 1 kHz loop on the asyncio event loop, so anything else that runs there delays the next torque command: a planner, loading a model, garbage collection, a thread holding the GIL. When commands are late, the robot stops with `communication_constraints_violation`. `NativeFrankaController` runs the same loop in C++, in a thread that never waits for Python. It has `FrankaController`'s constructor, methods and attributes, so porting legacy code means swapping the class:

```python
from aiofranka import NativeFrankaController, RobotInterface

robot = RobotInterface("172.16.0.2")
controller = NativeFrankaController(robot)  # instead of FrankaController(robot)
await controller.start()
controller.switch("osc")
await controller.set("ee_desired", target)
```

Server mode runs it too: `FrankaRemoteController`, `aiofranka start-server` and `aiofranka.start()` start the server with `NativeServerController`, and `aiofranka home`, `aiofranka gravcomp` and `aiofranka tool identify` run `NativeFrankaController`. Where the native loop is not built, the server, `home` and `gravcomp` fall back to the Python loop with a warning.

Its impedance, pid, osc and torque laws are ports of `FrankaController`'s. Stepped in lockstep from the same state, impedance, pid and torque mode send bit-identical torques, and the OSC agrees to 1e-12 Nm. On a simulated robot driven through libfranka's `readOnce()`/`writeOnce()`, the longest gap between two torque commands was:

| While Python... | `FrankaController` | `NativeFrankaController` |
|---|---|---|
| idles | 1.3 ms | 1.2 ms |
| blocks the event loop for 300 ms | 302 ms | 1.3 ms |
| runs a thread that holds the GIL for 300 ms | 18.8 ms | 1.2 ms |

What changes:

- A subclass's `step()` would never run, so `NativeFrankaController` refuses one: write it as a control law, and log every cycle with `record()` (both below).
- Attributes the loop reads (`kp`, `q_desired`, `ee_desired`, ...) are views of its memory. Assigning one copies the value in, and the loop takes it whole at its next cycle.
- While the loop runs, `robot.data` and `robot.robot_state` follow it, updated from the event loop about every millisecond. In simulation, the loop owns the simulated arm.
- On Linux, the loop's thread runs at SCHED_FIFO priority 80, which needs an rtprio limit (`ulimit -r`) of at least that. Set `controller.realtime_priority = 0` before `start()` for normal priority.
- The thread starts on the CPUs of the thread that calls `start()`. Set `controller.realtime_cpu` before `start()` to pin it to a CPU of its own, and keep Python's threads at normal priority: another SCHED_FIFO thread of the same priority on that CPU would hold the loop off until it yields.

On an FR3 driven from a Linux PREEMPT_RT laptop, tracking 2 cm OSC circles for 15 s per test, the native loop sent every command in time while the same process blocked its event loop for 300 ms every second, ran 4 threads holding the GIL, ran 20-thread BLAS, or collected garbage over a 3 M object heap: 0 robot states missed, the robot's command success rate never below 0.98. The Python loop dropped to 0.92 with no load and stopped with `communication_constraints_violation` under the BLAS load. Saturating every core of that laptop, even at nice 19, stalled its networking and stopped either loop; with the robot NIC's interrupt cores left free, the native loop was unaffected.

The loop is a compiled extension. The wheels for macOS on Apple Silicon (CPython 3.10 to 3.14) and Linux x86_64 (CPython 3.10 to 3.12) include it; elsewhere, or from a git checkout, it is built with a C++17 compiler. For a development install:

```bash
pip install "pybind11>=3.1,<3.2"
pip install --no-build-isolation -e .
```

It uses pylibfranka's libfranka and must be rebuilt for another libfranka minor version. Built with other pybind11 internals than pylibfranka (e.g. the official Linux pylibfranka 0.21.2, built with pybind11 3.0), it updates `robot.robot_state` in place instead of replacing it.

### Recording every cycle

`controller.record()` logs every cycle of the loop. After sending the torques, the loop writes the chosen fields into a buffer in C++ without waiting for Python, and the controller moves them to Python about every 10 ms, so a blocked event loop loses nothing:

```python
with controller.record(["time", "q", "dq", "tau", "tau_J_d"], path="control.npz") as recording:
    await run_policy(controller)
data = recording.data()  # {"time": (n,), "q": (n, 7), ...}, one row per cycle
```

The fields are those of the cycle (`cycle`, `time`, `wall_time`, `busy`, `q`, `dq`, `ee`, `tcp`, `jac`, `mm`, `last_torque`, `tau`, `robot_mode`), the controller's attributes as the cycle used them (`q_desired`, `ee_desired`, `kp`, ..., and the parameters of registered control laws), and every number of the robot state (`tau_J`, `tau_J_d`, `tau_ext_hat_filtered`, `O_F_ext_hat_K`, `control_command_success_rate`, ...; NaN in simulation). Without fields, `record()` takes `aiofranka.native.RECORD_FIELDS`.

- `tau` is the torque sent, after the rate limit and clip. The robot echoes each command it got in the `tau_J_d` of its next state, rounded to float32, so the rows show which commands arrived.
- `wall_time` is the host's `time.time()` when the robot state arrived, and `busy` the seconds from then to the command.
- With a `path`, `stop()` saves the rows there, as does a loop that stops with an error before the process exits. `recording.save(path)` saves the rows so far.
- The buffer holds `seconds` of cycles (default 60). If Python does not take the rows for longer, the loop drops new ones and `recording.dropped` counts them.

### Custom control laws

Write a new law in Python. aiofranka compiles it with Numba (`pip install "aiofranka[native]"`), and the loop calls it every millisecond without Python. It reads the state `s` (`qpos`, `qvel`, `ee`, `jac`, `mm`, `last_torque`, `time`, `dt`) and the controller's attributes `p`, keeps memory `m` between cycles, and fills the torques `tau`:

```python
from aiofranka import control_law

@control_law(params={"stiffness": 7, "damping": 7}, memory={"integral": 7})
def pi_damping(s, p, m, tau):
    error = p.q_desired - s.qpos
    m.integral[:] += error * s.dt
    tau[:] = p.stiffness * error + 2.0 * m.integral - p.damping * s.qvel

controller.switch(pi_damping)            # compiles it, a few seconds the first time
controller.stiffness = np.full(7, 60.0)  # its params are controller attributes, zero at first
controller.damping = np.full(7, 4.0)
await controller.set("q_desired", target)
```

Laws use numpy and math only (Numba's nopython mode). With `clip` (the default), the loop rate-limits and clips their torques like the built-in laws. Return a nonzero integer to stop the loop with an error. Try a law in simulation first: `controller.step()` runs one cycle at a time.

## License

aiofranka's original code is available under the MIT License; see [LICENSE](LICENSE). The bundled FR3 model and meshes retain their upstream Apache-2.0 and BSD-3-Clause terms in [aiofranka/model/LICENSE](aiofranka/model/LICENSE).

## Citation

If you use this library in your research, please cite:

```bibtex
@software{aiofranka,
  author = {Park, Younghyo},
  title = {aiofranka: Asyncio-based Franka Robot Control},
  year = {2025},
  url = {https://github.com/younghyopark/aiofranka}
}
```

## Acknowledgments

- Built on [libfranka](https://frankarobotics.github.io/docs/) by Franka Emika
- Uses [MuJoCo](https://mujoco.org/) physics engine
- Trajectory generation with [Ruckig](https://github.com/pantor/ruckig)
