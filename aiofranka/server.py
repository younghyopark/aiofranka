"""
aiofranka control server — runs the 1kHz control loop in a dedicated process.

Launch via CLI:  aiofranka start [--ip IP] [--foreground]

Features:
  - Auto unlock/lock brakes via Franka Desk API
  - Writes error state to shared memory so clients detect failures
  - Always releases control token on shutdown (crash or clean)
"""

import asyncio
import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid

import msgpack
import mujoco
import numpy as np
import zmq

from aiofranka.controller import FrankaController
from aiofranka.ipc import (
    StateBlock,
    zmq_endpoint_for_ip,
    pid_file_for_ip,
    progress_file_for_ip,
    STATUS_STOPPED,
    STATUS_ERROR,
    STATUS_RUNNING,
)
from aiofranka.robot import RobotInterface, set_macos_control_thread_qos

logger = logging.getLogger("aiofranka.server")

# Gain attributes that can be set remotely
_GAIN_ATTRS = {
    "kp", "kd", "ki", "ee_kp", "ee_kd", "null_kp", "null_kd",
    "torque_diff_limit", "torque_limit", "clip",
}

# Target attributes that can be set remotely
_TARGET_ATTRS = {
    "q_desired", "ee_desired", "torque",
}


class ServerController(FrankaController):
    """FrankaController subclass that writes state to shared memory each step.

    Overrides _run() to NOT sys.exit(1) on error — instead sets running=False
    and stores the error so the server can handle restart.
    """

    def __init__(self, robot: RobotInterface, shm_block: StateBlock):
        super().__init__(robot)
        self._shm = shm_block
        self._last_error = None

        # Pre-allocate reusable buffers to avoid per-iteration allocations
        model = robot.model
        self._ee_buf = np.eye(4)
        self._jac_buf = np.zeros((6, 7))
        self._mm_buf = np.zeros((7, 7))
        self._qpos_buf = np.empty(model.nq)
        self._qvel_buf = np.empty(model.nv)
        self._ctrl_buf = np.empty(model.nu)
        self._zero_torque = np.zeros(7)

    def step(self):
        robot = self.robot
        model = robot.model
        data = robot.data

        # Sync MuJoCo with real robot state (readOnce blocks for 1kHz tick)
        if robot.real:
            robot_state, _ = robot.torque_controller.readOnce()
            data.qpos[:] = robot_state.q
            data.qvel[:] = robot_state.dq
            data.ctrl[:] = robot_state.tau_J_d
            mujoco.mj_forward(model, data)

        # Build state using pre-allocated buffers (no per-iteration allocation)
        self._ee_buf[:3, :3] = data.site(robot.site_id).xmat.reshape(3, 3)
        self._ee_buf[:3, 3] = data.site(robot.site_id).xpos

        self._jac_buf[:] = 0
        mujoco.mj_jacSite(model, data, self._jac_buf[:3], self._jac_buf[3:], robot.site_id)

        self._mm_buf[:] = 0
        mujoco.mj_fullM(model, data, self._mm_buf)

        np.copyto(self._qpos_buf, data.qpos)
        np.copyto(self._qvel_buf, data.qvel)
        np.copyto(self._ctrl_buf, data.ctrl)

        self.state = {
            "qpos": self._qpos_buf,
            "qvel": self._qvel_buf,
            "ee": self._ee_buf,
            "jac": self._jac_buf,
            "mm": self._mm_buf,
            "last_torque": self._ctrl_buf,
        }

        # Run control law
        if self.type == "impedance":
            self._impedance_step(self.state)
        elif self.type == "pid":
            self._pid_step(self.state)
        elif self.type == "osc":
            self._osc_step(self.state)
        elif self.type == "torque":
            self._torque_step(self.state)

        # Write to shared memory
        full_state = {
            "qpos": self._qpos_buf, "qvel": self._qvel_buf, "ee": self._ee_buf,
            "jac": self._jac_buf, "mm": self._mm_buf, "last_torque": self._ctrl_buf,
            "q_desired": self.q_desired,
            "ee_desired": self.ee_desired,
            "torque": getattr(self, "torque", self._zero_torque),
            "initial_qpos": self.initial_qpos,
            "initial_ee": self.initial_ee,
        }
        self._shm.write_state(full_state)
        self._shm.write_ctrl_type(self.type)

    async def start(self):
        """Override start() to run robot.start() in a thread with timeout.

        pylibfranka's start_torque_control() can hang or segfault after a
        reflex error. Running it in a thread prevents it from killing the
        entire server process.
        """
        loop = asyncio.get_event_loop()
        logger.info("Starting torque control...")
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self.robot.start), timeout=10.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("robot.start() timed out (10s)")

        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(1)
        return self.task

    async def _run(self):
        """Override _run to NOT sys.exit(1) — let the server handle recovery."""
        if sys.platform == "darwin":
            # macOS has no CPU pinning or SCHED_FIFO. libfranka busy-waits for robot
            # states instead, which needs this thread on a performance core.
            if set_macos_control_thread_qos():
                logger.info("Control loop set to QoS USER_INTERACTIVE")
        else:
            # Pin to last CPU core for cache locality
            n_cpus = os.cpu_count() or 1
            rt_core = n_cpus - 1
            try:
                os.sched_setaffinity(0, {rt_core})
                logger.info(f"Control loop pinned to CPU {rt_core}")
            except Exception as e:
                logger.warning(f"Could not pin to CPU {rt_core}: {e}")
            # Elevate to SCHED_FIFO real-time priority
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
                logger.info("Control loop set to SCHED_FIFO priority 80")
            except PermissionError:
                logger.warning("Could not set SCHED_FIFO (no permission) — run with CAP_SYS_NICE")
            except Exception as e:
                logger.warning(f"Could not set SCHED_FIFO: {e}")

        self.running = True
        try:
            while self.running:
                t0 = time.time()
                self.step()
                dt = time.time() - t0
                if not self.robot.real:
                    await asyncio.sleep(1/1000. - dt)
                else:
                    await asyncio.sleep(0)
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
            # Do NOT sys.exit — return so server can retry


class CommandHandler:
    """Handles ZMQ REQ/REP commands from clients."""

    def __init__(self, controller: ServerController, shm_block: StateBlock, robot_ip: str):
        self.controller = controller
        self.shm = shm_block
        self.robot_ip = robot_ip
        self._active_moves = {}  # move_id -> {"task": Task, "done": bool, "error": str|None}
        self._should_stop = False  # set True on "stop" command to prevent retry
        self._loop = None  # set before starting the thread

    def run(self):
        """Run ZMQ command handler in a dedicated thread (never blocks the asyncio control loop)."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 500)  # 500ms timeout so we can check _should_stop
        endpoint = zmq_endpoint_for_ip(self.robot_ip)

        # Clean up stale socket file
        sock_path = endpoint.replace("ipc://", "")
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        sock.bind(endpoint)
        logger.info(f"ZMQ REP bound to {endpoint}")

        try:
            while not self._should_stop:
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue

                try:
                    msg = msgpack.unpackb(raw, raw=False)
                    resp = self._handle(msg)
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}

                sock.send(msgpack.packb(resp, use_bin_type=True))
        finally:
            sock.close()
            ctx.term()

    def _handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")

        if cmd == "status":
            return {
                "ok": True,
                "running": self.controller.running,
                "controller_type": self.controller.type,
                "ip": self.robot_ip,
                "error": self.controller._last_error,
            }

        elif cmd == "switch":
            ctrl_type = msg["type"]
            self.controller.switch(ctrl_type)
            return {"ok": True}

        elif cmd == "set":
            attr = msg["attr"]
            value = msg["value"]
            if isinstance(value, (bytes, bytearray)):
                value = np.frombuffer(value, dtype=np.float64).copy()
                shape = msg.get("shape")
                if shape:
                    value = value.reshape(shape)
            if attr in _TARGET_ATTRS or attr in _GAIN_ATTRS:
                with self.controller.state_lock:
                    setattr(self.controller, attr, value)
                return {"ok": True}
            else:
                return {"ok": False, "error": f"Unknown attribute: {attr}"}

        elif cmd == "set_freq":
            self.controller.set_freq(msg["freq"])
            return {"ok": True}

        elif cmd == "move":
            qpos = msg["qpos"]
            if isinstance(qpos, (bytes, bytearray)):
                qpos = np.frombuffer(qpos, dtype=np.float64).copy()
            elif isinstance(qpos, list):
                qpos = np.array(qpos)
            move_id = str(uuid.uuid4())[:8]
            self._active_moves[move_id] = {"task": None, "done": False, "error": None}
            future = asyncio.run_coroutine_threadsafe(self._do_move(move_id, qpos), self._loop)
            self._active_moves[move_id]["task"] = future
            return {"ok": True, "move_id": move_id}

        elif cmd == "move_status":
            move_id = msg["move_id"]
            info = self._active_moves.get(move_id)
            if info is None:
                return {"ok": False, "error": f"Unknown move_id: {move_id}"}
            return {"ok": True, "done": info["done"], "error": info["error"]}

        elif cmd == "stop":
            self.controller.running = False
            self._should_stop = True
            return {"ok": True}

        elif cmd == "initialize":
            self.controller.initialize()
            return {"ok": True}

        else:
            return {"ok": False, "error": f"Unknown command: {cmd}"}

    async def _do_move(self, move_id: str, qpos):
        try:
            await self.controller.move(qpos)
            self._active_moves[move_id]["done"] = True
        except Exception as e:
            self._active_moves[move_id]["done"] = True
            self._active_moves[move_id]["error"] = str(e)


def _write_pid(robot_ip: str):
    pid_path = pid_file_for_ip(robot_ip)
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))


def _remove_pid(robot_ip: str):
    pid_path = pid_file_for_ip(robot_ip)
    try:
        os.unlink(pid_path)
    except FileNotFoundError:
        pass


def _write_progress(robot_ip: str, step: int, total: int, label: str,
                     warn: str = "", **extra):
    """Write structured progress for the CLI to display."""
    import json as _json
    path = progress_file_for_ip(robot_ip)
    data = {"step": step, "total": total, "label": label}
    if warn:
        data["warn"] = warn
    data.update(extra)
    with open(path, "w") as f:
        _json.dump(data, f)


def _cleanup_ipc(robot_ip: str):
    """Remove socket file, PID file, and progress file."""
    sock_path = zmq_endpoint_for_ip(robot_ip).replace("ipc://", "")
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    try:
        os.unlink(progress_file_for_ip(robot_ip))
    except FileNotFoundError:
        pass
    _remove_pid(robot_ip)


# ── Desk API error classification ────────────────────────────────────────

class DeskAPIError(RuntimeError):
    """Base Desk API error with status code and error code."""
    def __init__(self, status_code, code="", message=""):
        self.status_code = status_code
        self.code = code
        super().__init__(message)

class AuthExpiredError(DeskAPIError):
    """401 -- authentication expired or invalid."""
    pass

class TokenInvalidError(DeskAPIError):
    """423 or 400 -- control token is wrong or missing."""
    pass

class SystemStateError(DeskAPIError):
    """424 -- system state prevents the operation."""
    pass

class TransientError(DeskAPIError):
    """503 -- transient error, retrying may succeed."""
    pass


class _DeskClient:
    """Franka Desk API client. Uses old API for login/token, new API for arm control."""

    def __init__(self, robot_ip: str, username: str, password: str, protocol: str = "https"):
        import requests
        requests.packages.urllib3.disable_warnings()
        self._session = requests.Session()
        self._session.verify = False
        self._base = f"{protocol}://{robot_ip}"
        self._username = username
        self._password = password
        self._token = None
        self._basic_auth = None  # for new API endpoints

    def _req(self, method, path, **kwargs):
        kwargs.setdefault("timeout", 30)
        return self._session.request(method, f"{self._base}{path}", **kwargs)

    def _ok(self, r):
        return r.status_code in (200, 204)

    def login(self):
        import base64, hashlib
        # Old API login (cookie-based, SHA256 password) — needed for token management
        bs = ','.join([str(b) for b in hashlib.sha256(
            f'{self._password}#{self._username}@franka'.encode('utf-8')).digest()])
        encoded = base64.encodebytes(bs.encode('utf-8')).decode('utf-8')
        r = self._req("POST", "/admin/api/login",
                       json={"login": self._username, "password": encoded})
        if r.status_code == 200:
            self._session.cookies.set("authorization", r.text)
        else:
            raise RuntimeError(f"Login failed (status {r.status_code}: {r.text})")
        # Also set Basic Auth header for new API endpoints (unlock/lock/FCI)
        self._basic_auth = f"Basic {base64.b64encode(f'{self._username}:{self._password}'.encode()).decode()}"
        logger.info("Logged in")

    def take_token(self):
        # Check if someone else already holds the token BEFORE requesting
        r0 = self._req("GET", "/admin/api/control-token")
        if self._ok(r0):
            info = r0.json()
            active = info.get("activeToken")
            if active is not None:
                owner = active.get("requestedBy", "unknown")
                raise RuntimeError(
                    f"Control token held by '{owner}' (via Franka Desk web GUI). "
                    f"Ask them to release control, or open {self._base} in your "
                    f"browser and click the 'Release Control' button."
                )

        r = self._req("POST", "/admin/api/control-token/request",
                      json={"requestedBy": self._username})
        if not self._ok(r):
            raise RuntimeError(f"Failed to take control token (status {r.status_code}: {r.text})")
        data = r.json()
        self._token = data.get("token")
        token_id = data.get("id")

        # Verify our token is actually active (catch stale token edge case)
        r2 = self._req("GET", "/admin/api/control-token")
        if self._ok(r2):
            info = r2.json()
            active_id = info.get("activeToken", {}).get("id")
            if active_id is not None and active_id != token_id:
                raise RuntimeError(
                    f"Stale control token (id={active_id}) held by another session. "
                    f"Reboot the robot to clear it: curl -k -X POST "
                    f"{self._base}/api/system:reboot -u user:pass "
                    f"-H 'Content-Type: application/json' -d '{{}}'"
                )
        logger.info("Acquired control token")

    def release_token(self):
        if self._token is None:
            return
        self._req("DELETE", "/admin/api/control-token",
                  json={"token": self._token})
        self._token = None
        logger.info("Released control token")

    def force_take_token(self):
        """Take control token, bypassing the active-token pre-check.

        Used to recover from stale tokens left by crashed server processes.
        The POST request may override the stale token on some firmware versions.
        """
        r = self._req("POST", "/admin/api/control-token/request",
                      json={"requestedBy": self._username})
        if not self._ok(r):
            raise RuntimeError(
                f"Cannot force-acquire control token (status {r.status_code}: {r.text})"
            )

        data = r.json()
        self._token = data.get("token")
        if not self._token:
            raise RuntimeError(
                f"Force-acquire returned no token (response: {data}). "
                f"Release it manually via the Franka Desk web GUI."
            )
        logger.info("Force-acquired control token")

    def _headers(self):
        """Headers for new API endpoints: Basic Auth + control token."""
        h = {}
        if self._token:
            h["X-Control-Token"] = self._token
        if self._basic_auth:
            h["Authorization"] = self._basic_auth
        return h

    def recover_errors(self):
        """Check for and confirm any safety errors that need recovery."""
        r = self._req("GET", "/api/safety/recovery", headers=self._headers())
        if self._ok(r):
            data = r.json()
            if data:
                logger.info(f"Safety recovery needed: {data}")
                error_type = data.get("type")
                if error_type:
                    r2 = self._req("POST", "/api/safety/recovery:confirm",
                                   json={"type": error_type},
                                   headers=self._headers())
                    if self._ok(r2):
                        logger.info("Safety error confirmed/recovered")
                        import time as _time
                        _time.sleep(2)

    def unlock(self):
        # New API endpoint with Basic Auth
        r = self._req("POST", "/api/arm/joints:unlock", headers=self._headers())
        if not self._ok(r):
            # Fall back to old API
            r = self._req("POST", "/desk/api/robot/open-brakes",
                          files={"force": "true"}, headers={"X-Control-Token": self._token})
        if not self._ok(r):
            raise RuntimeError(f"Unlock failed (status {r.status_code}: {r.text})")
        logger.info("Robot unlocked")

    def lock(self):
        r = self._req("POST", "/api/arm/joints:lock", headers=self._headers())
        if not self._ok(r):
            r = self._req("POST", "/desk/api/robot/close-brakes",
                          files={"force": "true"}, headers={"X-Control-Token": self._token})
        if not self._ok(r):
            raise RuntimeError(f"Lock failed (status {r.status_code}: {r.text})")
        logger.info("Robot locked")

    def activate_fci(self):
        r = self._req("POST", "/api/fci:activate", headers=self._headers())
        if not self._ok(r):
            r = self._req("POST", "/admin/api/control-token/fci",
                          json={"token": self._token})
        if not self._ok(r):
            raise RuntimeError(f"FCI activation failed (status {r.status_code}: {r.text})")
        logger.info("FCI activated")

    def is_fci_active(self) -> bool:
        """Check if FCI is currently active (regardless of who holds the token)."""
        r = self._req("GET", "/admin/api/control-token")
        if self._ok(r):
            return r.json().get("fciActive", False)
        return False

    def are_joints_unlocked(self) -> bool:
        """Check if all joints are unlocked (brakes open)."""
        r = self._req("GET", "/api/arm/joints", headers=self._headers())
        if self._ok(r):
            joints = r.json()
            return all(j.get("brakeStatus") != "Locked" for j in joints)
        return False

    def get_self_test_status(self) -> dict:
        """Get self-test status. Returns {'status': 'OK'|'Warning'|'Elapsed'|'Running', 'remaining': seconds}."""
        r = self._req("GET", "/api/safety/self-tests", headers=self._headers())
        if self._ok(r):
            return r.json()
        return {}

    def execute_self_tests(self):
        """Execute self-tests (blocking). Joints are locked during and re-unlocked after."""
        r = self._req("POST", "/api/safety/self-tests:execute", headers=self._headers(),
                       timeout=300)  # self-tests can take a few minutes
        if not self._ok(r):
            raise RuntimeError(f"Self-test execution failed (status {r.status_code}: {r.text})")

    def get_configuration(self) -> dict:
        """Get system configuration (end-effector, network, etc.)."""
        r = self._req("GET", "/api/configuration", headers=self._headers())
        if self._ok(r):
            return r.json()
        return {}

    def get_token_info(self) -> dict:
        """Get control token info (who holds it, FCI active, etc.)."""
        r = self._req("GET", "/admin/api/control-token")
        if self._ok(r):
            return r.json()
        return {}

    def set_configuration(self, config: dict):
        """PATCH system configuration. Only provided top-level keys are changed."""
        r = self._req("PATCH", "/api/configuration", json=config, headers=self._headers())
        if not self._ok(r):
            raise RuntimeError(f"Set configuration failed (status {r.status_code}: {r.text})")

    def deactivate_fci(self):
        self._req("POST", "/api/fci:deactivate", headers=self._headers())


class _DeskClientV2:
    """Franka Desk API client using the new REST API exclusively.

    Key differences from _DeskClient (old API):
    - Stateless Basic Auth on every request (no login/session cookies to expire)
    - control-token:take supports a ``timeout`` parameter for contention handling
    - Stores ``token_id`` for validation without releasing the token
    - Typed exceptions (TransientError, TokenInvalidError, etc.) enable smart retry
    - Built-in retry on 503 (transient) errors
    """

    def __init__(self, robot_ip: str, username: str, password: str,
                 protocol: str = "https"):
        import requests
        import base64
        requests.packages.urllib3.disable_warnings()
        self._session = requests.Session()
        self._session.verify = False
        self._base = f"{protocol}://{robot_ip}"
        self._username = username
        self._password = password
        self._token = None
        self._token_id = None
        # Stateless Basic Auth -- no login() needed, no session expiry
        self._basic_auth = (
            f"Basic {base64.b64encode(f'{username}:{password}'.encode()).decode()}"
        )

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _req(self, method, path, **kwargs):
        kwargs.setdefault("timeout", 30)
        return self._session.request(method, f"{self._base}{path}", **kwargs)

    def _headers(self):
        """Headers for all API requests: Basic Auth + control token."""
        h = {"Authorization": self._basic_auth}
        if self._token:
            h["X-Control-Token"] = self._token
        return h

    def _check(self, r, context=""):
        """Check response status, raise typed exception on error."""
        if r.status_code in (200, 204):
            return r
        # Try to parse error body
        code = ""
        message = ""
        try:
            body = r.json()
            code = body.get("code", "")
            message = body.get("message", "")
        except Exception:
            message = r.text
        msg = f"{context}: {r.status_code} {code} {message}".strip(": ")
        if r.status_code == 401:
            raise AuthExpiredError(401, code, msg)
        elif r.status_code == 423:
            raise TokenInvalidError(423, code, msg)
        elif r.status_code == 400 and "Control-Token" in (code + message):
            raise TokenInvalidError(400, code, msg)
        elif r.status_code == 424:
            raise SystemStateError(424, code, msg)
        elif r.status_code == 503:
            raise TransientError(503, code, msg)
        else:
            raise DeskAPIError(r.status_code, code, msg)

    def _with_retry(self, fn, max_retries=3, backoff=1.0, context=""):
        """Call fn(), retry on TransientError (503) or SystemStateError (424) with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except (TransientError, SystemStateError) as exc:
                if attempt == max_retries:
                    raise
                sleep_time = backoff * (2 ** attempt)
                logger.warning(
                    f"{context}: {exc.status_code} transient, retry {attempt + 1}/{max_retries} "
                    f"in {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)

    # ── Token management (new API) ──────────────────────────────────

    def take_token(self, timeout=10):
        """Take the control token via the new API.

        If someone else holds the token, blocks up to ``timeout`` seconds.
        On success, stores self._token and self._token_id.
        """
        body = {"owner": self._username}
        if timeout is not None:
            body["timeout"] = timeout
        r = self._req("POST", "/api/system/control-token:take",
                       json=body, headers=self._headers())
        self._check(r, "take_token")
        data = r.json()
        self._token = data["token"]
        self._token_id = data.get("tokenId")
        logger.info(f"Acquired control token (tokenId={self._token_id})")

    def release_token(self, *, best_effort: bool = False):
        """Release the control token.

        Args:
            best_effort: If True, log warnings instead of raising on failure.
                         If False (default), raise on non-success response.
        """
        if self._token is None:
            return
        r = self._req("POST", "/api/system/control-token:release",
                       headers=self._headers())
        if r.status_code in (200, 204):
            logger.info("Released control token")
            self._token = None
            self._token_id = None
        elif best_effort:
            logger.warning(f"release_token: {r.status_code} {r.text}")
            self._token = None
            self._token_id = None
        else:
            raise RuntimeError(
                f"Failed to release control token "
                f"(status {r.status_code}: {r.text})"
            )

    def validate_token(self) -> bool:
        """Check if our token is still the active one (without releasing it).

        This is the key primitive for retry: we can check token validity
        without releasing it and risking the web GUI race.
        """
        if self._token is None or self._token_id is None:
            return False
        r = self._req("GET", "/api/system/control-token", headers=self._headers())
        if r.status_code == 200:
            data = r.json()
            return data.get("tokenId") == self._token_id
        return False

    # ── Safety recovery ─────────────────────────────────────────────

    def recover_errors(self):
        """Check for and confirm any safety errors that need recovery."""
        def _do():
            r = self._req("GET", "/api/safety/recovery", headers=self._headers())
            self._check(r, "recover_errors (GET)")
            data = r.json()
            if data:
                logger.info(f"Safety recovery needed: {data}")
                error_type = data.get("type")
                if error_type:
                    r2 = self._req("POST", "/api/safety/recovery:confirm",
                                   json={"type": error_type},
                                   headers=self._headers())
                    self._check(r2, "recover_errors (confirm)")
                    logger.info("Safety error confirmed/recovered")
                    time.sleep(2)
        self._with_retry(_do, context="recover_errors")

    def can_auto_recover(self) -> bool:
        """Check if the current safety error can be auto-recovered.

        Returns False for JointLimitViolation and JointPositionError
        which require physical intervention (enabling device / safety-operator).
        """
        r = self._req("GET", "/api/safety/recovery", headers=self._headers())
        if r.status_code != 200:
            return False
        data = r.json()
        if not data:
            return True  # No recovery needed
        error_type = data.get("type", "")
        return error_type not in ("JointLimitViolation", "JointPositionError")

    # ── Arm control (same new API endpoints as _DeskClient) ─────────

    def unlock(self):
        """Unlock joints (open brakes)."""
        def _do():
            r = self._req("POST", "/api/arm/joints:unlock",
                           headers=self._headers())
            if r.status_code not in (200, 204):
                # Fall back to old endpoint
                r = self._req("POST", "/desk/api/robot/open-brakes",
                               files={"force": "true"},
                               headers={"X-Control-Token": self._token})
            self._check(r, "unlock")
            logger.info("Robot unlocked")
        self._with_retry(_do, context="unlock")

    def lock(self):
        """Lock joints (close brakes)."""
        def _do():
            r = self._req("POST", "/api/arm/joints:lock",
                           headers=self._headers())
            if r.status_code not in (200, 204):
                r = self._req("POST", "/desk/api/robot/close-brakes",
                               files={"force": "true"},
                               headers={"X-Control-Token": self._token})
            self._check(r, "lock")
            logger.info("Robot locked")
        self._with_retry(_do, context="lock")

    def activate_fci(self):
        """Activate the Franka Control Interface."""
        def _do():
            r = self._req("POST", "/api/fci:activate", headers=self._headers())
            if r.status_code not in (200, 204):
                r = self._req("POST", "/admin/api/control-token/fci",
                               json={"token": self._token})
            self._check(r, "activate_fci")
            logger.info("FCI activated")
        self._with_retry(_do, context="activate_fci")

    def deactivate_fci(self):
        self._req("POST", "/api/fci:deactivate", headers=self._headers())

    def is_fci_active(self) -> bool:
        """Check if FCI is currently active."""
        r = self._req("GET", "/api/fci", headers=self._headers())
        if r.status_code == 200:
            data = r.json()
            return data.get("status") == "Active"
        # Fall back to old endpoint
        r = self._req("GET", "/admin/api/control-token")
        if r.status_code == 200:
            return r.json().get("fciActive", False)
        return False

    def are_joints_unlocked(self) -> bool:
        """Check if all joints are unlocked (brakes open)."""
        r = self._req("GET", "/api/arm/joints", headers=self._headers())
        if r.status_code == 200:
            joints = r.json()
            return all(j.get("brakeStatus") != "Locked" for j in joints)
        return False

    def get_self_test_status(self) -> dict:
        r = self._req("GET", "/api/safety/self-tests", headers=self._headers())
        if r.status_code == 200:
            return r.json()
        return {}

    def execute_self_tests(self):
        def _do():
            r = self._req("POST", "/api/safety/self-tests:execute",
                           headers=self._headers(), timeout=300)
            self._check(r, "execute_self_tests")
        self._with_retry(_do, context="execute_self_tests")

    def get_configuration(self) -> dict:
        r = self._req("GET", "/api/configuration", headers=self._headers())
        if r.status_code == 200:
            return r.json()
        return {}

    def set_configuration(self, config: dict):
        r = self._req("PATCH", "/api/configuration", json=config,
                       headers=self._headers())
        self._check(r, "set_configuration")

    # ── System state / operating mode ────────────────────────────────

    def get_system_state(self) -> dict:
        """Get full system state (status, operatingMode, controlSerialNumber, cloud)."""
        r = self._req("GET", "/api/system", headers=self._headers())
        if r.status_code == 200:
            return r.json()
        return {}

    def get_operating_mode(self) -> str:
        """Get the current operating mode.

        Returns one of: "Execution", "Programming", "SafetyRecovery",
        "SelfTest", "Undefined".
        """
        r = self._req("GET", "/api/system/operating-mode",
                       headers=self._headers())
        if r.status_code == 200:
            return r.json().get("status", "")
        return ""

    def change_operating_mode(self, mode: str = "Execution"):
        """Change the operating mode to "Execution" or "Programming" (needs the control token).

        The Desk API only changes to Execution. Programming goes through the endpoint of
        the Desk web UI's mode switch, which takes the UI's login cookie.
        """
        def _do():
            if mode == "Execution":
                r = self._req("POST", "/api/system/operating-mode:change",
                               json={"desiredOperatingMode": mode},
                               headers=self._headers())
            else:
                r = self._req("POST", f"/desk/api/operating-mode/{mode.lower()}",
                               headers={"X-Control-Token": self._token},
                               cookies={"authorization": self._login_cookie()})
            self._check(r, "change_operating_mode")
            logger.info(f"Operating mode changed to {mode}")
        self._with_retry(_do, context="change_operating_mode")

    def _login_cookie(self) -> str:
        """Log in as the Desk web UI does, for the endpoints only the UI uses."""
        from aiofranka.client import FrankaClient
        r = self._req("POST", "/admin/api/login",
                       json={"login": self._username,
                             "password": FrankaClient._encode_password(self._username, self._password)})
        self._check(r, "login")
        return r.text

    # ── Arm info ─────────────────────────────────────────────────────

    def get_arm_info(self) -> dict:
        """Get arm information (connection status and type).

        Returns e.g. {"status": "Connected", "armType": "FR3"}.
        """
        r = self._req("GET", "/api/arm", headers=self._headers())
        if r.status_code == 200:
            return r.json()
        return {}


def _unlock_robot(robot_ip: str, username: str = "admin", password: str = "admin",
                  protocol: str = "https", on_progress=None,
                  run_self_tests: bool = False):
    """Unlock brakes and activate FCI via the new Desk API.

    on_progress(step, label) is called before each sub-step so the CLI
    can display numbered progress.
    """
    client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
    try:
        if on_progress:
            on_progress(1, "Logging in to Franka Desk")
        # No login() needed -- _DeskClientV2 uses stateless Basic Auth

        if on_progress:
            on_progress(2, "Acquiring control token")
        # New API's timeout handles contention (no force_take_token needed)
        client._with_retry(lambda: client.take_token(timeout=15),
                           context="take_token")

        client.recover_errors()

        step = 3
        if run_self_tests:
            if on_progress:
                on_progress(step, "Running self-tests")
            logger.info("Self-tests are due, executing...")
            client.execute_self_tests()
            logger.info("Self-tests completed")
            step += 1

        # FCI only activates in Execution, e.g. not after hand guiding in Programming
        if client.get_operating_mode() == "Programming":
            client.change_operating_mode("Execution")

        if on_progress:
            on_progress(step, "Unlocking joints")
        client.unlock()

        if on_progress:
            on_progress(step + 1, "Activating FCI")
        client.activate_fci()

        logger.info("Robot unlocked and FCI activated")
        return client
    except Exception as e:
        # Release token so it doesn't get stuck on next start
        try:
            client.release_token(best_effort=True)
        except Exception:
            pass
        logger.error(f"Failed to unlock robot: {e}")
        raise


_DEFAULT_HOME = [0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853]

async def _run_server(robot_ip: str, unlock: bool = True,
                      username: str = "admin", password: str = "admin",
                      protocol: str = "https", skip_token: bool = False,
                      lock_on_error: bool = False, home: bool = True,
                      controller_cls=None):
    """Main server coroutine with auto-unlock and retry on errors."""
    logger.info(f"Starting aiofranka server for {robot_ip}")

    # Total steps: 5 with unlock (login, token, unlock, fci, control loop), 1 without
    # +1 if self-tests need to run, +1 if homing
    total_steps = 5 if unlock else 1
    if home:
        total_steps += 1  # add homing step

    _progress_extra = {}
    def progress(step, label, warn="", **extra):
        if warn:
            _progress_extra["warn"] = warn
        _progress_extra.update(extra)
        _write_progress(robot_ip, step, total_steps, label, **_progress_extra)

    # Create shared memory + PID early so CLI can read errors
    shm = StateBlock(robot_ip, create=True)
    _write_pid(robot_ip)

    # Unlock brakes + activate FCI
    lock_client = None
    self_test_info = {}
    if unlock:
        # Pre-check: is the robot already unlocked + FCI active (e.g., via webGUI)?
        # Also check self-test status early.
        pre_ready = False
        self_test_info = {}
        try:
            probe = _DeskClientV2(robot_ip, username, password, protocol=protocol)
            pre_ready = probe.are_joints_unlocked() and probe.is_fci_active()
            self_test_info = probe.get_self_test_status()
        except Exception:
            pass

        self_test_due = self_test_info.get("status") == "Elapsed"

        if pre_ready:
            # Robot is already set up — skip unlock, just try to get token
            logger.info("Robot already unlocked with FCI active (pre-check)")
            progress(1, "Logging in to Franka Desk")
            if skip_token:
                logger.info("Skipping token acquisition (user chose to proceed without)")
                progress(2, "Acquiring control token", warn="no_token")
            else:
                progress(2, "Acquiring control token")
                try:
                    probe.take_token(timeout=15)
                    lock_client = probe
                except Exception as e:
                    logger.warning(
                        f"Could not acquire control token ({e}), but robot is "
                        f"already unlocked with FCI active. Proceeding without "
                        f"lock control (will NOT lock on shutdown)."
                    )
                    progress(2, "Acquiring control token", warn="no_token")

            # Run self-tests if due and we have the token
            if self_test_due and lock_client is not None:
                total_steps = 6
                progress(3, "Running self-tests")
                try:
                    lock_client.execute_self_tests()
                    logger.info("Self-tests completed")
                    # Joints may need re-unlock after self-tests
                    if not lock_client.are_joints_unlocked():
                        lock_client.unlock()
                    if not lock_client.is_fci_active():
                        lock_client.activate_fci()
                    self_test_info = lock_client.get_self_test_status()
                except Exception as e:
                    logger.warning(f"Self-test execution failed: {e}")
                progress(4, "Unlocking joints (already unlocked)")
                progress(5, "Activating FCI (already active)")
            else:
                progress(3, "Unlocking joints (already unlocked)")
                progress(4, "Activating FCI (already active)")
        else:
            # Robot needs full unlock — check if self-tests bump step count
            if self_test_due:
                total_steps = 6

            try:
                lock_client = _unlock_robot(
                    robot_ip, username, password, protocol=protocol,
                    on_progress=progress, run_self_tests=self_test_due,
                )
            except Exception as e:
                # Unlock failed — re-check in case robot became ready in the meantime
                try:
                    fallback = _DeskClientV2(robot_ip, username, password, protocol=protocol)
                    joints_ok = fallback.are_joints_unlocked()
                    fci_ok = fallback.is_fci_active()
                except Exception:
                    joints_ok = fci_ok = False

                if joints_ok and fci_ok:
                    logger.warning(
                        f"Could not acquire control token ({e}), but robot is "
                        f"already unlocked with FCI active. Proceeding without "
                        f"lock control (will NOT lock on shutdown)."
                    )
                    progress(2, "Acquiring control token", warn="no_token")
                    progress(3, "Unlocking joints (already unlocked)")
                    progress(4, "Activating FCI (already active)")
                    # lock_client stays None — skip lock/unlock on shutdown
                else:
                    missing = []
                    if not joints_ok:
                        missing.append("joints are locked")
                    if not fci_ok:
                        missing.append("FCI is not active")
                    err = f"{e}\n\nCannot proceed: {', '.join(missing)}."
                    shm.write_error(err)
                    # Keep shm + pid alive so the CLI can read the error
                    import time as _time
                    _time.sleep(10)
                    shm.close()
                    shm.unlink()
                    _cleanup_ipc(robot_ip)
                    return

            # Refresh self-test status after unlock (may have been run)
            if lock_client:
                try:
                    self_test_info = lock_client.get_self_test_status()
                except Exception:
                    pass

        # Store self-test remaining hours for CLI display
        remaining_sec = self_test_info.get("remaining", 0)
        if remaining_sec:
            _progress_extra["self_test_remaining"] = remaining_sec

    # Save token to disk (unified pipeline — server + standalone use same file)
    if lock_client is not None and lock_client._token is not None:
        _save_token_state(robot_ip, lock_client._token, lock_client._token_id)

    # Safety net: release token on atexit (works for foreground mode)
    # and via signal handler (works for daemon mode where os._exit skips atexit)
    def _release_token_safe():
        if lock_client is not None:
            try:
                lock_client.release_token(best_effort=True)
            except Exception:
                pass
    atexit.register(_release_token_safe)

    progress(total_steps - 1 if home else total_steps, "Starting 1kHz control loop")

    # Set up signal handlers
    shutdown_requested = False

    def shutdown_handler():
        nonlocal shutdown_requested
        logger.info("Received shutdown signal")
        shutdown_requested = True
        controller.running = False
        # Don't release token here — the finally block needs it for
        # deactivate_fci/lock. If we release early, the Franka Desk
        # web GUI grabs it before we can lock.

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    # Start ZMQ command handler in a dedicated thread so it never blocks the
    # asyncio control loop (libfranka requires strict 1kHz timing).
    if controller_cls is None:
        controller_cls = ServerController
    elif controller_cls == "native":
        from aiofranka.server_native import NativeServerController
        controller_cls = NativeServerController
    robot = RobotInterface(robot_ip, read_tool=False)
    controller = controller_cls(robot, shm)
    cmd_handler = CommandHandler(controller, shm, robot_ip)
    cmd_handler._loop = asyncio.get_event_loop()
    cmd_thread = threading.Thread(target=cmd_handler.run, daemon=True, name="zmq-cmd-handler")
    cmd_thread.start()

    max_retries = 3

    try:
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    logger.info("Starting control loop")
                    ctrl_task = await controller.start()

                    # Move to default pose and hold
                    if home:
                        progress(total_steps, "Moving to home pose")
                        logger.info("Moving to home pose...")
                        controller.kp = np.ones(7) * 80
                        controller.kd = np.ones(7) * 4
                        await controller.move(_DEFAULT_HOME)
                        logger.info("Home pose reached, holding with kp=80 kd=4")
                else:
                    logger.info(
                        f"Retry {attempt}/{max_retries}: reinitializing control loop..."
                    )
                    progress(total_steps,
                             f"Recovering (attempt {attempt}/{max_retries})")

                    # 1. Stop old robot connection (best-effort)
                    try:
                        robot.stop()
                    except Exception:
                        pass

                    # 2. Validate/reacquire token (without releasing — avoids
                    #    the web GUI race described in the old signal handler comment)
                    if lock_client is not None:
                        if not lock_client.validate_token():
                            logger.warning("Token invalid, reacquiring...")
                            lock_client._with_retry(
                                lambda: lock_client.take_token(timeout=15),
                                context="take_token (reacquire)")
                            _save_token_state(
                                robot_ip, lock_client._token,
                                lock_client._token_id,
                            )

                        # 3. Check if we can auto-recover the safety error
                        if not lock_client.can_auto_recover():
                            logger.error(
                                "Safety error requires physical intervention "
                                "(JointLimitViolation or JointPositionError). "
                                "Cannot auto-recover."
                            )
                            shm.write_error(
                                "Safety error requires manual recovery. "
                                "Use the enabling device or Franka Desk web GUI."
                            )
                            break

                        # 4. Recover safety errors
                        lock_client.recover_errors()

                        # 5. Re-unlock + re-activate FCI if needed
                        if not lock_client.are_joints_unlocked():
                            lock_client.unlock()
                        if not lock_client.is_fci_active():
                            lock_client.activate_fci()

                    # 6. Brief settle time for robot to stabilize
                    await asyncio.sleep(2)

                    # 7. New robot + controller (old C++ object may be in bad state)
                    robot = RobotInterface(robot_ip, read_tool=False)
                    controller = controller_cls(robot, shm)
                    cmd_handler.controller = controller  # update ZMQ handler ref

                    ctrl_task = await controller.start()
                    progress(total_steps, "Starting 1kHz control loop")

                shm.write_status(STATUS_RUNNING)
                await ctrl_task
                break  # clean exit (user stopped the controller)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Control loop error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                shm.write_error(str(e))
                if cmd_handler._should_stop or shutdown_requested:
                    break  # user requested stop, don't retry
                if "Reflex" in str(e):
                    logger.error("Robot is in Reflex mode — wait a few seconds and relaunch")
                    break
                if attempt == max_retries:
                    logger.error("Max retries reached, shutting down")
                    break
                await asyncio.sleep(3)  # backoff before retry

    finally:
        controller.running = False
        cmd_handler._should_stop = True
        cmd_thread.join(timeout=1.0)

        try:
            robot.stop()
        except Exception:
            pass

        # Decide whether to lock joints: always on clean shutdown (SIGTERM),
        # only if lock_on_error on error exits (crash, reflex, start failure, etc).
        should_lock = shutdown_requested or lock_on_error

        if lock_client is not None:
            if should_lock:
                stop_total = 4
                def stop_progress(step, label):
                    _write_progress(robot_ip, step, stop_total, label)

                stop_progress(1, "Stopping 1kHz control loop")

                stop_progress(2, "Deactivating FCI")
                try:
                    lock_client.deactivate_fci()
                except Exception as e:
                    logger.warning(f"Failed to deactivate FCI: {e}")

                stop_progress(3, "Locking joints")
                try:
                    lock_client.lock()
                except Exception as e:
                    logger.warning(f"Failed to lock robot: {e}")

                stop_progress(4, "Releasing control token")
                try:
                    lock_client.release_token(best_effort=True)
                except Exception as e:
                    logger.warning(f"Failed to release token: {e}")
            else:
                logger.info("Skipping lock on error exit (lock_on_error=False)")
                _write_progress(robot_ip, 1, 2, "Stopping 1kHz control loop")
                _write_progress(robot_ip, 2, 2, "Releasing control token (joints left unlocked)")
                try:
                    lock_client.release_token(best_effort=True)
                except Exception as e:
                    logger.warning(f"Failed to release token: {e}")
        else:
            _write_progress(robot_ip, 1, 1, "Stopping 1kHz control loop")

        # Only clear the saved token if we owned it (lock_client was set).
        # When unlock=False (e.g. start_subprocess), we don't own the token —
        # clearing it would prevent the parent from restarting.
        if lock_client is not None:
            _clear_token(robot_ip)
        shm.write_status(STATUS_STOPPED)
        shm.close()
        shm.unlink()
        sock_path = zmq_endpoint_for_ip(robot_ip).replace("ipc://", "")
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        _remove_pid(robot_ip)
        logger.info("Server shutdown complete")


def run_server(robot_ip: str, foreground: bool = False,
               unlock: bool = True, username: str = "admin", password: str = "admin",
               protocol: str = "https", skip_token: bool = False,
               lock_on_error: bool = False, home: bool = True,
               controller_cls=None):
    """Entry point for the server process."""
    if foreground:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
    else:
        config_dir = os.path.expanduser("~/.aiofranka")
        os.makedirs(config_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            filename=os.path.join(config_dir, "server.log"),
        )

    asyncio.run(_run_server(robot_ip, unlock=unlock, username=username,
                            password=password, protocol=protocol,
                            skip_token=skip_token, lock_on_error=lock_on_error,
                            home=home, controller_cls=controller_cls))


def daemonize_and_run(robot_ip: str, unlock: bool = True,
                      username: str = "admin", password: str = "admin",
                      protocol: str = "https", skip_token: bool = False,
                      lock_on_error: bool = False, home: bool = True,
                      controller_cls=None):
    """Fork into background and run the server."""
    pid = os.fork()
    if pid > 0:
        time.sleep(0.5)
        return

    os.setsid()

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    sys.stdin = open(os.devnull, 'r')
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    run_server(robot_ip, foreground=False, unlock=unlock,
               username=username, password=password, protocol=protocol,
               skip_token=skip_token, lock_on_error=lock_on_error, home=home,
               controller_cls=controller_cls)
    os._exit(0)  # Don't fall through to caller's code


def start_subprocess(ip: str, *,
                     timeout: float = 60.0,
                     controller_cls=None) -> "multiprocessing.Process":
    """Start server in a child process tied to the parent's lifecycle.

    Unlike daemonize_and_run(), the child process terminates when the parent
    dies (via PR_SET_PDEATHSIG on Linux and daemon=True on the Process).

    Args:
        ip: Robot IP address.
        timeout: Seconds to wait for the server to become ready.
        controller_cls: The server's controller class (default: ServerController), or
            "native" for NativeServerController, whose 1 kHz loop runs in C++.

    Returns:
        The multiprocessing.Process running the server.

    Raises:
        RuntimeError: If the server fails to start or times out.
    """
    import multiprocessing

    from aiofranka.ipc import StateBlock, STATUS_RUNNING, STATUS_ERROR, pid_file_for_ip

    ip, username, password = _resolve_from_config(ip, None, None, interactive=False)

    # Kill any existing server for this IP before starting a new one
    pid_path = pid_file_for_ip(ip)
    if os.path.exists(pid_path):
        with open(pid_path) as f:
            old_pid = int(f.read().strip())
        try:
            os.kill(old_pid, signal.SIGTERM)
            # Wait for it to die
            for _ in range(50):  # up to 5s
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
        # Own process group so Ctrl+C (SIGINT) from the terminal never
        # reaches us — _run_server re-registers SIGINT via asyncio, so
        # SIG_IGN alone isn't enough.  Parent sends SIGTERM for clean shutdown.
        os.setpgrp()
        # Auto-terminate when parent dies (Linux)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1
        except Exception:
            pass
        run_server(ip, foreground=True, unlock=False,
                   username=username, password=password,
                   lock_on_error=False, home=False,
                   controller_cls=controller_cls)

    proc = multiprocessing.Process(target=_target, daemon=True)
    proc.start()

    # Poll shared memory for readiness
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)

        # Check shared memory first (even if process died, error may be written)
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
                        bar = "█" * filled + "░" * (20 - filled)
                        sys.stdout.write(f"\r  [{bar}] {(i+1)*2}%")
                        sys.stdout.flush()
                    print()
                    return start_subprocess(ip, timeout=timeout, controller_cls=controller_cls)
                raise RuntimeError(err)
            shm.close()
        except FileNotFoundError:
            pass

        if not proc.is_alive():
            # Process died — try one more time to read the error
            try:
                shm = StateBlock(ip, create=False, track=False)
                err = shm.read_error()
                shm.close()
                if err:
                    raise RuntimeError(err)
            except FileNotFoundError:
                pass
            raise RuntimeError(
                "Server process exited unexpectedly. "
                "Check ~/.aiofranka/server.log for details."
            )

    proc.terminate()
    raise RuntimeError(f"Server did not become ready within {timeout}s")


# ── Gravity compensation mode ─────────────────────────────────────────────

async def _run_gravcomp_loop(robot_ip: str, damping: float = 0.0,
                             http_port: int = 0):
    """Run gravity compensation control loop until Ctrl+C.

    Assumes robot is already unlocked with FCI active.
    If http_port > 0, serves GET /qpos on that port returning JSON joint positions.
    """
    from aiofranka.controller import FrankaController

    robot = RobotInterface(robot_ip, read_tool=False)
    controller = FrankaController(robot)
    controller.kp = np.zeros(7)
    controller.kd = np.ones(7) * damping
    controller.ki = np.zeros(7)

    stop_event = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    # Optional HTTP server for querying qpos
    http_server = None
    if http_port > 0:
        async def _handle_http(reader, writer):
            try:
                request_line = await reader.readline()
                # Drain remaining headers
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break

                path = request_line.decode().split(" ")[1] if request_line else "/"

                if path == "/qpos":
                    state = getattr(controller, "state", None)
                    if state and "qpos" in state:
                        qpos = list(np.round(state["qpos"], 6))
                    else:
                        qpos = None
                    body = json.dumps({"qpos": qpos}).encode()
                    header = (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                        b"\r\n"
                    )
                else:
                    body = b'{"error": "not found, try GET /qpos"}'
                    header = (
                        b"HTTP/1.1 404 Not Found\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                        b"\r\n"
                    )
                writer.write(header + body)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()

        http_server = await asyncio.start_server(_handle_http, "0.0.0.0", http_port)
        print(f"  HTTP server listening on :{http_port}  →  GET /qpos")

    ctrl_task = await controller.start()

    # Wait until Ctrl+C or control loop exits
    done, _ = await asyncio.wait(
        [ctrl_task, asyncio.create_task(stop_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if http_server is not None:
        http_server.close()
        await http_server.wait_closed()

    controller.running = False
    try:
        robot.stop()
    except Exception:
        pass


def run_gravcomp_loop(robot_ip: str, damping: float = 0.0, http_port: int = 0):
    """Run gravity compensation control loop (blocks until Ctrl+C)."""
    asyncio.run(_run_gravcomp_loop(robot_ip, damping=damping, http_port=http_port))


_HOME_QPOS = [0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853]


async def _run_home_move(robot_ip: str):
    """Move robot to home position. Assumes robot is already unlocked with FCI active."""
    from aiofranka.controller import FrankaController

    robot = RobotInterface(robot_ip, read_tool=False)
    controller = FrankaController(robot)

    base = np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])
    controller.kp = base * 80
    controller.kd = base * 4
    controller.ki = np.zeros(7)

    await controller.start()
    await controller.move(_HOME_QPOS)

    controller.running = False
    try:
        robot.stop()
    except Exception:
        pass


def run_home_move(robot_ip: str):
    """Move robot to home position (blocks until done)."""
    asyncio.run(_run_home_move(robot_ip))


# ── Config helpers (shared with CLI) ───────────────────────────────────────

_CONFIG_DIR = os.path.expanduser("~/.aiofranka")
_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.json")


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(config: dict):
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _resolve_from_config(ip, username, password, interactive: bool = False):
    """Fill in ip/username/password from ~/.aiofranka/config.json when defaults are used.

    If interactive=True and credentials are not in config, prompt the user.
    """
    import getpass as _getpass
    config = _load_config()
    if ip is None:
        ip = config.get("last_ip", "172.16.0.2")
    if username is None:
        username = config.get("username")
    if password is None:
        password = config.get("password")

    if username is None or password is None:
        if interactive:
            print("  Robot web UI credentials (saved to ~/.aiofranka/config.json)")
            username = input("  Username [admin]: ").strip() or "admin"
            password = _getpass.getpass("  Password: ")
            config["username"] = username
            config["password"] = password
            _save_config(config)
        else:
            username = username or "admin"
            password = password or "admin"

    return ip, username, password


# ── Progress display for standalone operations ───────────────────────────

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _ansi(code: str) -> str:
    return code if _IS_TTY else ""


_BOLD = _ansi("\033[1m")
_DIM = _ansi("\033[2m")
_GREEN = _ansi("\033[32m")
_RED = _ansi("\033[31m")
_YELLOW = _ansi("\033[33m")
_RST = _ansi("\033[0m")


def _step_line(step: int, total: int, label: str, status: str) -> str:
    tag = f"{_DIM}[{step}/{total}]{_RST}"
    ndots = max(2, 38 - len(label))
    dots = " " + "." * ndots
    return f"  {tag} {label}{dots} {status}"


import threading as _threading


def _run_with_spinner(label: str, step: int, total: int, fn, *args, **kwargs):
    """Run fn() in a thread, showing a spinner on the current step.

    On success, prints the step as done and returns the result.
    On failure, prints the step as failed and re-raises.
    """
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = _threading.Thread(target=_target, daemon=True)
    t.start()

    spin_idx = 0
    while t.is_alive():
        frame = _SPINNER[spin_idx % len(_SPINNER)]
        line = _step_line(step, total, label, f"{_YELLOW}{frame}{_RST}")
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()
        t.join(timeout=0.1)
        spin_idx += 1

    if error[0] is not None:
        line = _step_line(step, total, label, f"{_RED}failed{_RST}")
        sys.stdout.write(f"\r{line}\n")
        sys.stdout.flush()
        raise error[0]

    line = _step_line(step, total, label, f"{_GREEN}done{_RST}")
    sys.stdout.write(f"\r{line}\n")
    sys.stdout.flush()
    return result[0]


# ── Programmatic API ──────────────────────────────────────────────────────

def start(ip: str = None, *, foreground: bool = False,
          unlock: bool = True, username: str = None, password: str = None,
          protocol: str = "https", lock_on_error: bool = False,
          timeout: float = 60.0, native: bool = False) -> int:
    """Start the aiofranka server from a Python script.

    When ip, username, or password are not provided, values are read from
    ~/.aiofranka/config.json (same config the CLI uses). If no config exists,
    the user is prompted interactively.

    Args:
        ip: Robot IP address (default: from config or 172.16.0.2).
        foreground: If True, blocks and runs in the current process.
        unlock: Auto-unlock joints and activate FCI.
        username: Franka Desk web UI username (default: from config, or prompts).
        password: Franka Desk web UI password (default: from config, or prompts).
        protocol: "http" or "https".
        lock_on_error: If True, lock joints when the server dies due to a control error.
            If False (default), joints are left unlocked on error so you can recover.
        timeout: Seconds to wait for the server to become ready (ignored if foreground).
        native: Run the 1 kHz control loop in C++ (NativeServerController).

    Returns:
        The server PID (0 if foreground, since it blocks).

    Raises:
        RuntimeError: If the server fails to start within the timeout or encounters an error.
    """
    from aiofranka.ipc import StateBlock, STATUS_RUNNING, STATUS_ERROR, pid_file_for_ip

    ip, username, password = _resolve_from_config(ip, username, password, interactive=True)

    # Release any saved standalone token (from a previous unlock()) since
    # the server will acquire its own token.
    saved_token, saved_token_id = _load_token_state(ip)
    if saved_token is not None:
        try:
            client = _DeskClientV2(ip, username, password, protocol=protocol)
            client._token = saved_token
            client._token_id = saved_token_id
            client.release_token(best_effort=True)
        except Exception:
            pass
        _clear_token(ip)

    # Check if already running
    pid_path = pid_file_for_ip(ip)
    if os.path.exists(pid_path):
        with open(pid_path) as f:
            old_pid = int(f.read().strip())
        try:
            os.kill(old_pid, 0)
            return old_pid  # already running
        except ProcessLookupError:
            os.unlink(pid_path)

    controller_cls = "native" if native else None
    if foreground:
        run_server(ip, foreground=True, unlock=unlock,
                   username=username, password=password, protocol=protocol,
                   lock_on_error=lock_on_error, controller_cls=controller_cls)
        return 0

    daemonize_and_run(ip, unlock=unlock, username=username, password=password,
                      protocol=protocol, lock_on_error=lock_on_error,
                      controller_cls=controller_cls)

    # Wait for server to become ready
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)
        try:
            shm = StateBlock(ip, create=False, track=False)
            status = shm.read_status()
            if status == STATUS_RUNNING:
                shm.close()
                with open(pid_path) as f:
                    return int(f.read().strip())
            if status == STATUS_ERROR:
                err = shm.read_error()
                shm.close()
                raise RuntimeError(f"Server failed to start: {err}")
            shm.close()
        except FileNotFoundError:
            pass

    raise RuntimeError(f"Server did not become ready within {timeout}s")


def stop(ip: str = None, *, timeout: float = 30.0) -> None:
    """Stop the aiofranka server from a Python script.

    Args:
        ip: Robot IP address (default: from config or 172.16.0.2).
        timeout: Seconds to wait for the server to exit.

    Raises:
        RuntimeError: If no server is running or it fails to stop in time.
    """
    from aiofranka.ipc import pid_file_for_ip

    ip, _, _ = _resolve_from_config(ip, None, None)
    pid_path = pid_file_for_ip(ip)
    if not os.path.exists(pid_path):
        raise RuntimeError(f"No server running for {ip}")

    with open(pid_path) as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        os.unlink(pid_path)
        raise RuntimeError(f"No server running for {ip} (stale PID file removed)")

    os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # Clean up PID file if daemon didn't
            try:
                os.unlink(pid_path)
            except FileNotFoundError:
                pass
            return

    raise RuntimeError(f"Server (PID {pid}) did not stop within {timeout}s")


def _token_file_for_ip(ip: str) -> str:
    """Path to saved control token file for standalone unlock/lock."""
    safe_ip = ip.replace(".", "_")
    return os.path.join(_CONFIG_DIR, f"token_{safe_ip}.json")


def _save_token(ip: str, token: str):
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_token_file_for_ip(ip), "w") as f:
        json.dump({"token": token}, f)


def _load_token(ip: str) -> str | None:
    try:
        with open(_token_file_for_ip(ip)) as f:
            return json.load(f).get("token")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _clear_token(ip: str):
    try:
        os.unlink(_token_file_for_ip(ip))
    except FileNotFoundError:
        pass


def _save_token_state(ip: str, token: str, token_id: int | None = None):
    """Save token + token_id to disk (unified pipeline)."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_token_file_for_ip(ip), "w") as f:
        json.dump({"token": token, "token_id": token_id}, f)


def _load_token_state(ip: str) -> tuple[str | None, int | None]:
    """Load token + token_id from disk. Returns (token, token_id)."""
    try:
        with open(_token_file_for_ip(ip)) as f:
            data = json.load(f)
            return data.get("token"), data.get("token_id")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None, None


def _is_server_running(ip: str) -> bool:
    """Check if an aiofranka server is running for this IP."""
    from aiofranka.ipc import pid_file_for_ip
    pid_path = pid_file_for_ip(ip)
    if not os.path.exists(pid_path):
        return False
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        try:
            os.unlink(pid_path)
        except FileNotFoundError:
            pass
        return False


def lock(ip: str = None, *, username: str = None, password: str = None,
         protocol: str = "https") -> None:
    """Deactivate FCI, lock the robot joints (close brakes), and release the control token.

    If a saved token exists (from a previous unlock()), it is reused.
    If a server is running, raises RuntimeError — use stop() instead.

    Args:
        ip: Robot IP address (default: from config or 172.16.0.2).
        username: Franka Desk web UI username (default: from config, or prompts).
        password: Franka Desk web UI password (default: from config, or prompts).
        protocol: "http" or "https".
    """
    ip, username, password = _resolve_from_config(ip, username, password, interactive=True)
    if _is_server_running(ip):
        raise RuntimeError(
            f"Server is running for {ip}. Use aiofranka.stop() instead "
            f"(it will lock joints automatically)."
        )

    total = 4
    print()

    client = _DeskClientV2(ip, username, password, protocol=protocol)

    # Step 1: Acquire control token (reuse saved token if valid)
    saved_token, saved_token_id = _load_token_state(ip)
    if saved_token is not None:
        client._token = saved_token
        client._token_id = saved_token_id
        if not client.validate_token():
            # Saved token is stale — release it so take_token won't deadlock
            logger.warning("Saved token is invalid, releasing and acquiring new one...")
            try:
                client.release_token(best_effort=True)
            except Exception:
                pass
            _clear_token(ip)
            client._token = None
            client._token_id = None

    if client._token is None:
        _run_with_spinner("Acquiring control token", 1, total,
                          lambda: client._with_retry(
                              lambda: client.take_token(timeout=15),
                              context="take_token"))
    else:
        line = _step_line(1, total, "Acquiring control token",
                          f"{_GREEN}done{_RST} {_DIM}(reused){_RST}")
        print(line)

    try:
        # Step 2: Deactivate FCI
        _run_with_spinner("Deactivating FCI", 2, total, client.deactivate_fci)

        # Step 3: Locking joints
        _run_with_spinner("Locking joints", 3, total, client.lock)

        # Step 4: Releasing token
        _run_with_spinner("Releasing control token", 4, total,
                          client.release_token)
        _clear_token(ip)
    except Exception:
        try:
            client.release_token(best_effort=True)
            _clear_token(ip)
        except Exception:
            pass
        raise

    print(f"\n  {_GREEN}Locked{_RST}\n")


def unlock(ip: str = None, *, username: str = None, password: str = None,
           protocol: str = "https") -> None:
    """Unlock the robot joints (open brakes) and activate FCI.

    It first recovers safety errors, runs the self-tests if they are overdue and
    switches from Programming to Execution, where FCI runs.

    The control token is kept (saved to disk) so FCI stays active.
    Call lock() to deactivate FCI, lock joints, and release the token.

    If a server is running, this is a no-op (robot is already unlocked).

    Args:
        ip: Robot IP address (default: from config or 172.16.0.2).
        username: Franka Desk web UI username (default: from config, or prompts).
        password: Franka Desk web UI password (default: from config, or prompts).
        protocol: "http" or "https".
    """
    ip, username, password = _resolve_from_config(ip, username, password, interactive=True)
    if _is_server_running(ip):
        return  # server already has it unlocked

    total = 4
    print()

    client = _DeskClientV2(ip, username, password, protocol=protocol)

    # Step 1: Acquire control token (reuse saved token if valid)
    saved_token, saved_token_id = _load_token_state(ip)
    if saved_token is not None:
        client._token = saved_token
        client._token_id = saved_token_id
        if not client.validate_token():
            # Saved token is stale — release it so take_token won't deadlock
            logger.warning("Saved token is invalid, releasing and acquiring new one...")
            try:
                client.release_token(best_effort=True)
            except Exception:
                pass
            _clear_token(ip)
            client._token = None
            client._token_id = None

    # Check self-test status before starting
    self_test_due = False
    try:
        st_info = client.get_self_test_status()
        self_test_due = st_info.get("status") == "Elapsed"
    except Exception:
        pass

    if self_test_due:
        total += 1  # extra step for self-tests

    # FCI only activates in Execution: switch back from Programming (hand guiding)
    programming = client.get_operating_mode() == "Programming"
    if programming:
        total += 1  # extra step for the switch

    if client._token is None:
        _run_with_spinner("Acquiring control token", 1, total,
                          lambda: client._with_retry(
                              lambda: client.take_token(timeout=15),
                              context="take_token"))
    else:
        line = _step_line(1, total, "Acquiring control token",
                          f"{_GREEN}done{_RST} {_DIM}(reused){_RST}")
        print(line)

    try:
        # Step 2: Recover safety errors (if any)
        _run_with_spinner("Recovering safety errors", 2, total,
                          client.recover_errors)

        step = 3

        # Run self-tests if overdue (must happen before unlock)
        if self_test_due:
            _run_with_spinner("Running self-tests (overdue)", step, total,
                              client.execute_self_tests)
            step += 1

        if programming:
            _run_with_spinner("Switching to Execution", step, total,
                              client.change_operating_mode, "Execution")
            step += 1

        # Unlock joints
        _run_with_spinner("Unlocking joints", step, total, client.unlock)

        # Activate FCI
        _run_with_spinner("Activating FCI", step + 1, total, client.activate_fci)

        # Save token + token_id to disk so lock() can pick it up later
        _save_token_state(ip, client._token, client._token_id)
    except Exception:
        # On failure, release the token so it doesn't get stuck
        try:
            client.release_token(best_effort=True)
            _clear_token(ip)
        except Exception:
            pass
        raise

    print(f"\n  {_GREEN}Unlocked{_RST} {_DIM}(FCI active){_RST}\n")


def set_configuration(
    ip: str = None,
    *,
    mass: float = None,
    com: list[float] = None,
    inertia: list[float] = None,
    translation: list[float] = None,
    rotation: list[float] = None,
    ee_name: str = None,
    username: str = None,
    password: str = None,
    protocol: str = "https",
) -> dict:
    """Set end-effector configuration on the robot via the Franka Desk API.

    Uses the saved control token from a previous unlock(). Only the parameters
    you provide are updated; omitted parameters are left unchanged.

    Args:
        ip: Robot IP address (default: from config or 172.16.0.2).
        mass: End-effector mass in kg.
        com: Center of mass [x, y, z] in meters.
        inertia: Inertia matrix as [x11, x12, x13, x22, x23, x33] in kg*m^2.
        translation: F_T_EE translation [x, y, z] in meters.
        rotation: F_T_EE rotation [roll, pitch, yaw] in radians.
        ee_name: End-effector type: "None", "FrankaHand", "CobotPump",
            "GenericCan" or "Other" (default: the current one, or "Other"
            when setting a mass without an end effector).
        username: Franka Desk web UI username.
        password: Franka Desk web UI password.
        protocol: "http" or "https".

    Returns:
        dict: The updated configuration from the robot.

    Raises:
        RuntimeError: If no saved token exists and token cannot be acquired,
            or if the configuration update fails.

    Example:
        >>> import aiofranka
        >>> aiofranka.unlock("172.16.0.2")
        >>> aiofranka.set_configuration(mass=0.5, com=[0, 0, 0.05])
        >>> aiofranka.lock("172.16.0.2")
    """
    ip, username, password = _resolve_from_config(ip, username, password, interactive=False)

    # Validate inputs early
    if com is not None and len(com) != 3:
        raise ValueError("com must be [x, y, z]")
    if inertia is not None and len(inertia) != 6:
        raise ValueError("inertia must be [x11, x12, x13, x22, x23, x33]")
    if translation is not None and len(translation) != 3:
        raise ValueError("translation must be [x, y, z]")
    if rotation is not None and len(rotation) != 3:
        raise ValueError("rotation must be [roll, pitch, yaw]")
    if all(v is None for v in (mass, com, inertia, translation, rotation, ee_name)):
        raise ValueError("At least one parameter must be provided")

    client = _DeskClientV2(ip, username, password, protocol=protocol)

    # Use saved token from a previous unlock()
    saved_token, saved_token_id = _load_token_state(ip)
    took_token = False
    if saved_token is not None:
        client._token = saved_token
        client._token_id = saved_token_id
        # Validate saved token is still active
        if not client.validate_token():
            logger.warning("Saved token is invalid, acquiring new one...")
            client._token = None
            client._token_id = None
            client._with_retry(lambda: client.take_token(timeout=15),
                               context="take_token")
            took_token = True
    else:
        client._with_retry(lambda: client.take_token(timeout=15),
                           context="take_token")
        took_token = True

    try:
        # Fetch current config so we can merge user values on top
        # (the API requires all fields to be present in params)
        current = client.get_configuration()
        cur_ee = current.get("endEffectorConfiguration", {}).get("endEffector", {})
        cur_params = cur_ee.get("params", {})

        if ee_name is None:
            ee_name = cur_ee.get("name", "Other")
            # A mass needs an end effector to belong to.
            if ee_name == "None" and mass:
                ee_name = "Other"

        # Merge: start from current, override with user-provided values
        merged = dict(cur_params)
        if mass is not None:
            merged["mass"] = mass
        if com is not None:
            merged["centerOfMass"] = {"x": com[0], "y": com[1], "z": com[2]}
        if inertia is not None:
            merged["inertia"] = {
                "x11": inertia[0], "x12": inertia[1], "x13": inertia[2],
                "x22": inertia[3], "x23": inertia[4], "x33": inertia[5],
            }
        cur_tf = merged.get("transformation", {})
        if translation is not None:
            cur_tf["translation"] = {"x": translation[0], "y": translation[1], "z": translation[2]}
        if rotation is not None:
            cur_tf["rotation"] = {"roll": rotation[0], "pitch": rotation[1], "yaw": rotation[2]}
        merged["transformation"] = cur_tf

        patch = {
            "endEffectorConfiguration": {
                "endEffector": {
                    "name": ee_name,
                    "params": merged,
                }
            }
        }
        client.set_configuration(patch)
        return client.get_configuration()
    finally:
        if took_token:
            client.release_token(best_effort=True)
