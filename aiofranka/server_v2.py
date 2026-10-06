"""
aiofranka v2 server — runs the 1kHz control loop in a dedicated RT thread
instead of asyncio for minimal jitter on PREEMPT_RT kernels.

The control loop runs in a tight thread that calls robot.step() (which blocks
for the next libfranka 1kHz tick on real robots) without any asyncio overhead.
For simulated robots, it spin-waits for precise 1ms timing.

Everything else (ZMQ command handler, startup, shutdown) reuses the original
server infrastructure.

Legacy: its loop runs in Python. The server's default, NativeServerController
(aiofranka.server_native), runs it in C++.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time

import mujoco
import numpy as np

from aiofranka.ipc import StateBlock, STATUS_RUNNING, STATUS_ERROR, pid_file_for_ip
from aiofranka.robot import RobotInterface, set_macos_control_thread_qos
from aiofranka.server import (
    ServerController, run_server,
    _resolve_from_config,
)

logger = logging.getLogger("aiofranka.server")


class ServerControllerV2(ServerController):
    """ServerController that runs the 1kHz loop in a dedicated thread.

    On real robots, robot.step() blocks for the next libfranka tick (~1ms),
    so no sleep is needed — the thread just calls step() in a tight loop.

    On simulated robots, spin-waits with time.perf_counter() for precise
    1ms timing (trades CPU for accuracy).
    """

    # Jitter monitoring thresholds (expected: 1.0ms per iteration)
    JITTER_WARN_LO_MS = 0.9   # warn when iteration is faster than this (ms)
    JITTER_WARN_HI_MS = 1.1   # warn when iteration is slower than this (ms)
    JITTER_ERROR_MS = 10.0    # error when iteration exceeds this (ms)
    JITTER_LOG_INTERVAL = 5.0  # summarize jitter stats every N seconds

    def __init__(self, robot: RobotInterface, shm_block: StateBlock):
        super().__init__(robot, shm_block)
        self._rt_thread = None

    async def start(self):
        """Start robot and launch the RT control thread."""
        loop = asyncio.get_event_loop()
        logger.info("Starting torque control (v2 RT thread)...")
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self.robot.start), timeout=10.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("robot.start() timed out (10s)")

        self.running = True
        self._rt_thread = threading.Thread(
            target=self._run_rt, daemon=True, name="rt-control-loop"
        )
        self._rt_thread.start()

        # Create an asyncio task that completes when the RT thread exits,
        # so the existing server infrastructure can await it.
        async def _wait_for_thread():
            while self._rt_thread.is_alive():
                await asyncio.sleep(0.1)

        self.task = asyncio.create_task(_wait_for_thread())
        await asyncio.sleep(1)
        return self.task

    def _run_rt(self):
        """The RT control loop — runs in a dedicated thread with per-phase profiling."""
        if sys.platform == "darwin":
            # macOS has no CPU pinning or SCHED_FIFO. libfranka busy-waits for robot
            # states instead, which needs this thread on a performance core.
            if set_macos_control_thread_qos():
                logger.info("RT thread set to QoS USER_INTERACTIVE")
        else:
            # Pin to last CPU core and set SCHED_FIFO for minimal jitter
            n_cpus = os.cpu_count() or 1
            rt_core = n_cpus - 1
            try:
                os.sched_setaffinity(0, {rt_core})
                logger.info(f"RT thread pinned to CPU {rt_core}")
            except Exception as e:
                logger.warning(f"Could not pin RT thread to CPU {rt_core}: {e}")
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
                logger.info("RT thread set to SCHED_FIFO priority 80")
            except PermissionError:
                logger.warning("Could not set SCHED_FIFO (no permission) — run with CAP_SYS_NICE for best RT performance")
            except Exception as e:
                logger.warning(f"Could not set SCHED_FIFO: {e}")

        last_t = time.perf_counter()
        jitter_log_t = last_t
        max_dt_ms = 0.0
        min_dt_ms = float('inf')
        window_warn = 0
        window_error = 0
        total_warn = 0
        total_error = 0
        iteration = 0
        is_real = self.robot.real

        # Profiling accumulators (in seconds)
        _PROFILE_PHASES = ["readOnce", "mj_fwd", "state_build", "ctrl_law", "writeOnce", "shm_write"]
        prof_sum = {k: 0.0 for k in _PROFILE_PHASES}
        prof_max = {k: 0.0 for k in _PROFILE_PHASES}
        prof_count = 0

        # Cache references for speed
        robot = self.robot
        model = robot.model
        data = robot.data
        site_id = robot.site_id
        tc = robot.torque_controller
        shm = self._shm

        # Pre-allocate reusable buffers to avoid per-iteration allocations
        _ee = np.eye(4)
        _jac = np.zeros((6, 7))
        _mm = np.zeros((7, 7))
        _qpos = np.empty(model.nq)
        _qvel = np.empty(model.nv)
        _ctrl = np.empty(model.nu)
        _zero_torque = np.zeros(7)

        try:
            while self.running:
                # === Phase 1: readOnce (blocking wait for next robot state) ===
                t1 = time.perf_counter()
                if is_real:
                    robot_state, _ = tc.readOnce()
                    t2 = time.perf_counter()

                    # === Phase 2: mj_forward (sync MuJoCo) ===
                    data.qpos[:] = robot_state.q
                    data.qvel[:] = robot_state.dq
                    data.ctrl[:] = robot_state.tau_J_d
                    mujoco.mj_forward(model, data)
                    t3 = time.perf_counter()
                else:
                    t2 = t1
                    t3 = t1

                # === Phase 3: build state dict (ee, jac, mm) ===
                _ee[:3, :3] = data.site(site_id).xmat.reshape(3, 3)
                _ee[:3, 3] = data.site(site_id).xpos

                _jac[:] = 0
                mujoco.mj_jacSite(model, data, _jac[:3], _jac[3:], site_id)

                _mm[:] = 0
                mujoco.mj_fullM(model, data, _mm)

                np.copyto(_qpos, data.qpos)
                np.copyto(_qvel, data.qvel)
                np.copyto(_ctrl, data.ctrl)

                state = {
                    "qpos": _qpos,
                    "qvel": _qvel,
                    "ee": _ee,
                    "jac": _jac,
                    "mm": _mm,
                    "last_torque": _ctrl,
                }
                self.state = state
                t4 = time.perf_counter()

                # === Phase 4: control law (compute torques) ===
                if self.type == "impedance":
                    self._impedance_step(state)
                elif self.type == "pid":
                    self._pid_step(state)
                elif self.type == "osc":
                    self._osc_step(state)
                elif self.type == "torque":
                    self._torque_step(state)
                t5 = time.perf_counter()
                # Note: ctrl_law includes writeOnce() since the control methods
                # call robot.step() internally. We measure writeOnce below
                # separately next iteration via readOnce timing.

                # === Phase 5: shared memory write ===
                full_state = {
                    "qpos": _qpos, "qvel": _qvel, "ee": _ee,
                    "jac": _jac, "mm": _mm, "last_torque": _ctrl,
                    "q_desired": self.q_desired,
                    "ee_desired": self.ee_desired,
                    "torque": getattr(self, "torque", _zero_torque),
                    "initial_qpos": self.initial_qpos,
                    "initial_ee": self.initial_ee,
                }
                shm.write_state(full_state)
                shm.write_ctrl_type(self.type)
                t6 = time.perf_counter()

                # === Accumulate profiling stats ===
                prof_count += 1
                phases = {
                    "readOnce": t2 - t1,
                    "mj_fwd": t3 - t2,
                    "state_build": t4 - t3,
                    "ctrl_law": t5 - t4,
                    "writeOnce": 0,  # included in ctrl_law (robot.step inside control methods)
                    "shm_write": t6 - t5,
                }
                for k, v in phases.items():
                    prof_sum[k] += v
                    if v > prof_max[k]:
                        prof_max[k] = v

                # --- jitter monitoring ---
                now = t6
                dt_ms = (now - last_t) * 1000.0
                last_t = now
                iteration += 1

                if iteration > 1:
                    if dt_ms > max_dt_ms:
                        max_dt_ms = dt_ms
                    if dt_ms < min_dt_ms:
                        min_dt_ms = dt_ms

                    if dt_ms > self.JITTER_ERROR_MS:
                        window_error += 1
                        total_error += 1
                    elif dt_ms < self.JITTER_WARN_LO_MS or dt_ms > self.JITTER_WARN_HI_MS:
                        window_warn += 1
                        total_warn += 1

                    # write cumulative jitter stats to shared memory for clients
                    shm.write_jitter_stats(max_dt_ms, total_warn, total_error)

                    # periodic summary + profile to server log
                    if now - jitter_log_t >= self.JITTER_LOG_INTERVAL:
                        if window_warn > 0 or window_error > 0:
                            logger.warning(
                                f"Jitter summary (last {self.JITTER_LOG_INTERVAL:.0f}s): "
                                f"{window_warn} warnings, {window_error} errors, "
                                f"dt range={min_dt_ms:.3f}-{max_dt_ms:.3f}ms"
                            )

                        # Print profile breakdown
                        if prof_count > 0:
                            total_us = sum(prof_sum.values()) / prof_count * 1e6
                            parts = []
                            for k in _PROFILE_PHASES:
                                avg_us = prof_sum[k] / prof_count * 1e6
                                max_us = prof_max[k] * 1e6
                                if avg_us > 0.1:  # skip zero phases
                                    parts.append(f"{k}={avg_us:.0f}/{max_us:.0f}us")
                            logger.info(
                                f"Profile (avg/max, n={prof_count}): "
                                f"{', '.join(parts)} | total={total_us:.0f}us"
                            )

                        jitter_log_t = now
                        max_dt_ms = 0.0
                        min_dt_ms = float('inf')
                        window_warn = 0
                        window_error = 0
                        prof_sum = {k: 0.0 for k in _PROFILE_PHASES}
                        prof_max = {k: 0.0 for k in _PROFILE_PHASES}
                        prof_count = 0

                if not is_real:
                    # Spin-wait for precise 1ms timing in simulation
                    target = t1 + 0.001
                    while time.perf_counter() < target:
                        pass

        except Exception as e:
            self.running = False
            self._last_error = str(e)
            logger.error(f"Control loop error: {self._last_error}")
            self._shm.write_error(self._last_error)
            if self.error_callback is not None:
                try:
                    self.error_callback(self._last_error)
                except Exception as cb_err:
                    logger.error(f"Error in error_callback: {cb_err}")

    async def _run(self):
        """Not used in v2 — control loop runs in _run_rt thread."""
        raise RuntimeError("ServerControllerV2 uses _run_rt, not _run")


def start_subprocess_v2(ip: str, *, timeout: float = 60.0) -> multiprocessing.Process:
    """Start a v2 server (RT-threaded control loop) in a child process.

    Same as start_subprocess() but uses ServerControllerV2 for the control loop.
    """
    ip, username, password = _resolve_from_config(ip, None, None, interactive=False)

    # Kill any existing server for this IP
    pid_path = pid_file_for_ip(ip)
    if os.path.exists(pid_path):
        with open(pid_path) as f:
            old_pid = int(f.read().strip())
        try:
            os.kill(old_pid, signal.SIGTERM)
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(old_pid, 0)
                except ProcessLookupError:
                    break
            else:
                os.kill(old_pid, signal.SIGKILL)
                time.sleep(0.2)
        except ProcessLookupError:
            pass
        if os.path.exists(pid_path):
            os.unlink(pid_path)

    def _target():
        os.setpgrp()
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        except Exception:
            pass
        run_server(ip, foreground=True, unlock=False,
                   username=username, password=password,
                   lock_on_error=False, home=False,
                   controller_cls=ServerControllerV2)

    proc = multiprocessing.Process(target=_target, daemon=True)
    proc.start()

    # Poll shared memory for readiness
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)

        try:
            shm = StateBlock(ip, create=False, track=False)
            status = shm.read_status()
            if status == STATUS_RUNNING:
                shm.close()
                return proc
            if status == STATUS_ERROR:
                err = shm.read_error()
                shm.close()
                proc.terminate()
                if "Reflex" in err:
                    import sys
                    print("Robot is in Reflex mode — waiting 5s before retry...")
                    for i in range(50):
                        time.sleep(0.1)
                        filled = (i + 1) * 20 // 50
                        bar = "\u2588" * filled + "\u2591" * (20 - filled)
                        sys.stdout.write(f"\r  [{bar}] {(i+1)*2}%")
                        sys.stdout.flush()
                    print()
                    return start_subprocess_v2(ip, timeout=timeout)
                raise RuntimeError(err)
            shm.close()
        except FileNotFoundError:
            pass

        if not proc.is_alive():
            try:
                shm = StateBlock(ip, create=False, track=False)
                err = shm.read_error()
                shm.close()
                if err:
                    raise RuntimeError(err)
            except FileNotFoundError:
                pass
            raise RuntimeError("Server process died before becoming ready")

    proc.terminate()
    raise RuntimeError(f"Server failed to start within {timeout}s")
