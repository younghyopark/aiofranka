# Changelog

## Unreleased

### Highlights

- The native loop is the default everywhere. `aiofranka start-server`, `aiofranka.start()` and `FrankaRemoteController` start the server with `NativeServerController`, `aiofranka home` and `aiofranka gravcomp` run `NativeFrankaController`, and the README, the docs and their examples use it. The Python loop is soft-deprecated: `FrankaController`, the server's `ServerController` (`aiofranka start-server --python`, `aiofranka.start(native=False)`, `FrankaRemoteController(native=False)`) and `FrankaRemoteControllerV2` keep working, but new code should use the native loop. Where it is not built, the server, `home` and `gravcomp` fall back to the Python loop with a warning. `start-server --native` and `FrankaRemoteControllerNative` stay for existing code.
- `NativeFrankaController.record()` logs every cycle of the native loop: the state, the targets and gains the cycle used, the torque sent and any number of the robot state, with the host time each state arrived. The loop writes the rows into a buffer in C++ without waiting for Python, so a blocked event loop loses none, and `stop()` returns them as numpy arrays or saves them to an .npz file, as does a loop that stops with an error.

### Changes

- `NativeFrankaController.realtime_cpu` pins the native loop's thread to a CPU on Linux. `NativeServerController` uses it instead of the last core when it is set.
- The system identification collectors `04_collect_joint_sysid.py` and `06_collect_osc_sysid.py` run on `NativeFrankaController`. They set the targets at the configuration's policy rate, as a policy would, and record every cycle with `record()`; what they share is in `examples/sysid.py`. Their plans are the same as before. `06` plans once, after connecting, with the payload in the model.
- `05_fit_joint_sysid.py` and `07_fit_osc_sysid.py` replay the target each cycle used rather than one per policy step from the window's start, so they fit recordings whose targets changed between policy steps. Earlier recordings replay as before.
- `aiofranka rt-benchmark` benchmarks the native loop (`NativeFrankaController`) by default, as `start()` runs it (at SCHED_FIFO priority 80 on Linux), and records every cycle with `record()`. It holds the current pose in gravcomp, impedance and OSC in turn and compares them; `--mode` picks the modes, and one mode prints the full report. `--python` benchmarks the Python loop instead, now with `FrankaController`'s laws, and keeps its per-phase breakdown; without the native extension it falls back to that. Like `tool identify`, it switches from Programming to Execution and refuses to run beside a server. `--v2`, which only changed the report's label, is gone.

## 0.7.1 - 2026-10-05

### Changes

- `aiofranka camera calibrate` switches the robot to Programming mode, where the arm moves by hand while the guiding button on the end effector is held, instead of running gravity compensation over FCI. It reads the joint positions from Desk, as its web UI does; `--damping` is gone, and the camera extra adds websockets.
- `aiofranka gravcomp --mode program` switches the robot to Programming mode, as `aiofranka mode program` does, where the arm moves by hand only while the guiding button on the end effector is held. The default, `--mode execute`, runs gravity compensation over FCI as before.
- `aiofranka tool identify` runs its 1 kHz loop in C++ (`NativeFrankaController`), and switches from Programming to Execution before activating FCI, as `aiofranka unlock` does.
- The collision checks of `tool identify` keep the tool 5 cm from the floor, as documented, instead of 10 cm: MuJoCo adds the margins of the two geoms.

## 0.7.0 - 2026-10-05

### Highlights

- Add `NativeFrankaController`, a drop-in for `FrankaController` whose 1 kHz control loop runs in C++, in a thread that never waits for Python: blocking the event loop or holding the GIL no longer delays torque commands. Its impedance, pid, osc and torque laws match `FrankaController`'s to rounding. For server mode, use `FrankaRemoteControllerNative` or `aiofranka start-server --native`.
- Write new control laws in Python with `aiofranka.control_law`: Numba compiles them for the native loop (`pip install "aiofranka[native]"`), and their parameters become controller attributes.

### Changes

- The wheels for macOS on Apple Silicon (CPython 3.10 to 3.14) and Linux x86_64 (CPython 3.10 to 3.12) include the native loop, the extension `aiofranka._native`. Elsewhere aiofranka installs without it; building from source compiles it with pybind11 3.1 and a C++17 compiler.
- Install on Linux with pip 25, which Python 3.12's venv brings: the macOS dependency's marker no longer compares `platform_release`, which pip 25 evaluated on Linux too and could not parse there. On Apple Silicon, aiofranka now needs macOS 15 or newer to install with its dependencies, as pylibfranka-macos does.
- Examples 00 to 03 use `NativeFrankaController`, and `examples/08_native_custom_law.py` holds the arm with a custom law. The system identification collectors (04, 06) keep `FrankaController`, whose `step()` they override.
- `RobotInterface` defers to a running native loop: `state`, `state_minimal` and `sync_mj()` take its last state, `step()` refuses, `stop()` stops it first, and `sync_payload()` gives it the new model.
- On Linux, the native loop's thread runs at SCHED_FIFO priority 80 (`realtime_priority`). `controller.loop_stats()` reports the loop's timing and what the robot saw: the largest gap between its states, the states missed and the lowest command success rate.

## 0.6.1 - 2026-10-03

### Changes

- `aiofranka mode program` switches to Programming to hand-guide the robot, as Desk's mode switch does: it deactivates FCI, opens the brakes if they are closed and hands the control token back to Desk. `aiofranka mode execute` switches back. They replace `aiofranka mode --set`.
- `aiofranka unlock` runs the self-tests when they are overdue, as `aiofranka.unlock()` does. Both, and the server, switch from Programming back to Execution before activating FCI.

## 0.6.0 - 2026-10-03

### Highlights

- Add `aiofranka camera calibrate` and `aiofranka camera fit` to locate a fixed camera relative to the robot with an AprilCube held on the flange: the arm is moved by hand in damped gravity compensation, views are captured by themselves whenever it rests at a new pose, and a terminal view shows the cube and the image regions still without a view. The fit writes `calibration.json` with `T_base_camera`, `T_ee_cube`, the intrinsics and held-out reprojection errors. Install with `pip install "aiofranka[camera]"`; the default cube is aprilcube's printable calibration cube.
- Manage Desk end-effector profiles as tools, from Python (`save_tool`, `load_tool`, `unload_tool`, `list_tools`, `remove_tool`) and the CLI (`aiofranka tool identify|load|unload|list|remove`). `aiofranka tool identify` and `controller.identify_payload()` estimate the tool's mass and center of mass from the joint torques at rest, approaching each pose from both sides. `aiofranka status` shows the active profile.
- Keep how a policy drives the robot in a controller configuration file: the mode, gains, null-space target, TCP, policy rate and the Desk tool it needs. `aiofranka.load_config()` reads and checks it, and `controller.activate()` applies it. It refuses if the tool is not the active Desk profile, or if the arm is farther than `arrival_tolerance` from the configuration's null-space target. `configs/` has examples, including system-identified OSC configurations of three Pocky sticks.
- `controller.set_tcp()` sets the tool center point the OSC controls, and `last_command` keeps the torque actually sent.
- Add system identification examples. `04_collect_joint_sysid.py` and `06_collect_osc_sysid.py` record steps, multisines and ramps with a policy's gains and rate. `05_fit_joint_sysid.py` and `07_fit_osc_sysid.py` fit the controller gains and each joint's armature, damping and friction loss in batched MuJoCo with CMA-ES, holding out one pose, and add the fit to the configuration's `sim` section.

### Changes

- `RobotInterface` clears any load set by an earlier connection, merges the payload the robot compensates into the MuJoCo model, and reads the active Desk profile once when it connects (read-only, 3 s timeout; the server skips it).
- `move()` ends on the exact target and waits until every joint is within `arrival_tolerance` (0.03 rad, at most 3 s). It plays its trajectory at 50 Hz whatever `set_freq()` says.
- Depend on pyyaml, and add the optional `camera` extra (aiocamera and aprilcube).
- The docs take their version from `pyproject.toml`. The repository moved to younghyopark/aiofranka.

## 0.5.1 - 2026-09-28

### Highlights

- On Apple Silicon Macs with macOS 15 or newer, depend on [pylibfranka-macos](https://pypi.org/project/pylibfranka-macos/), an unofficial macOS build of pylibfranka, so `pip install aiofranka` works without building pylibfranka from source.

## 0.5.0 - 2026-09-28

### Breaking changes

- Replace the examples with minimal async mode scripts. The trajectory collection, plotting, system-identification, SpaceMouse teleoperation, and Robotiq examples moved to the `research-scripts` branch.

### Highlights

- Support macOS on Apple Silicon. PyPI has no macOS build of pylibfranka, so on macOS aiofranka does not depend on it; install pylibfranka from [younghyopark/libfranka](https://github.com/younghyopark/libfranka/tree/macos-support) first, as described in the README. Without it, connecting to a robot, including starting the server, fails with the install command.
- On macOS, run every control loop (async mode, server, v2 RT thread, and `rt-benchmark`) with the QoS class `USER_INTERACTIVE`, which libfranka's busy-waiting needs to stay on a performance core, and skip the Linux-only CPU pinning, `SCHED_FIFO`, and `mlockall` instead of failing.
- Report the robot-side view in `rt-benchmark`: the response time from `readOnce` to `writeOnce` against a 300 us budget, skipped robot states, and dropped commands with their likely cause.
- Warm up the first `mj_jacSite`, `mj_fullM`, and `data.site()` calls when creating a `RobotInterface`, so the first control cycles no longer miss their deadline.
- Release the torque controller in `RobotInterface.stop()`, so the stopped motion is cleaned up right away instead of at interpreter exit.
- Remove the per-step torque-clipping diagnostic prints. Torque rate and torque limit clipping still apply.
- Add minimal examples that move a joint, hold with joint impedance, hold with OSC, and stream zero torque. Each runs in MuJoCo without a robot.

## 0.4.0 - 2026-08-22

### Breaking changes

- Require Python 3.10+ and MuJoCo 3.10+; Python 3.8 and 3.9 are no longer supported.
- `aiofranka gravcomp` now leaves the robot unlocked with FCI active after stopping. Run `aiofranka lock` when finished.
- Remove unused bundled cube and extrinsic sample assets.

### Highlights

- Restore clean-install compatibility with modern MuJoCo by migrating every mass-matrix calculation to the current `mj_fullM` API.
- Improve 1 kHz real-time control with preallocated buffers, CPU affinity and `SCHED_FIFO` support, jitter telemetry, second-generation remote/server controllers, and new benchmark and tuning tools.
- Make server shutdown, restart, and control-token recovery more reliable, with actionable server-failure reporting.
- Restore `aiofranka start-server` and add `home`, Robotiq `gripper`, and `rt-benchmark` commands, plus an optional gravity-compensation `/qpos` endpoint.
- Add synchronous background Robotiq control through `GripperRemoteController`.
- Add new trajectory collection, plotting, and system-identification examples.
