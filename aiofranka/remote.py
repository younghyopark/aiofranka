"""
Synchronous remote controller client for aiofranka.

Communicates with the aiofranka server process via shared memory (state reads)
and ZMQ (commands). Drop-in replacement for FrankaController with a sync API.
"""

import atexit
import contextlib
import json
import logging
import os
import time
import threading

import msgpack
import numpy as np
import zmq

from aiofranka.ipc import StateBlock, zmq_endpoint_for_ip, STATUS_ERROR

logger = logging.getLogger("aiofranka.remote")

DEFAULT_IP = "172.16.0.2"

# Attributes that get sent to server via ZMQ when assigned
_REMOTE_ATTRS = {
    "kp", "kd", "ki", "ee_kp", "ee_kd", "null_kp", "null_kd",
    "torque_diff_limit", "torque_limit", "clip",
    "q_desired", "ee_desired", "torque",
}

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RST = "\033[0m"


class ServerDiedError(RuntimeError):
    """Raised when the server process dies (e.g. reflex error).

    Attributes:
        server_error: The error message from the server (e.g. libfranka reflex details).
    """
    def __init__(self, server_error: str):
        self.server_error = server_error
        # Build a user-friendly message
        lines = [
            f"\n  {_RED}Server died:{_RST} {server_error}\n",
        ]
        if "reflex" in server_error.lower():
            lines.append(f"  The robot entered {_BOLD}Reflex{_RST} mode (safety stop).")
            lines.append(f"  This usually means the robot hit a joint/velocity/torque limit.\n")
        lines.append(f"  To recover:")
        lines.append(f"    1. Run {_BOLD}aiofranka gravcomp{_RST} to freely move the robot")
        lines.append(f"       to a safe configuration, then Ctrl+C and restart your script.")
        lines.append(f"    2. Or just restart your script (it will auto-recover if possible).\n")
        super().__init__("\n".join(lines))


class FrankaRemoteController:
    """
    Synchronous remote controller for Franka robots.

    Connects to a running aiofranka server process and provides the same
    API as FrankaController, but fully synchronous (no async/await needed).

    The 1kHz control loop runs in the server process, in C++ (NativeServerController),
    or in Python where the native loop is not built. This client sends
    commands via ZMQ and reads state from shared memory (zero-copy).

    Args:
        robot_ip (str): Robot IP address (default: the last used one)
        native (bool): False runs the server's legacy Python loop (ServerController)

    Examples:
        >>> controller = FrankaRemoteController()  # default IP
        >>> controller.start()
        >>> controller.switch("impedance")
        >>> controller.set_freq(50)
        >>> for i in range(100):
        ...     state = controller.state
        ...     controller.set("q_desired", target)
        >>> controller.stop()
    """

    # The server's controller class, for start_subprocess() (None: its default, the native loop).
    _server_controller = None

    def __init__(self, robot_ip: str = None, *, home: bool = True, native: bool = True):
        if robot_ip is None:
            robot_ip = _load_last_ip()
        self.robot_ip = robot_ip
        self._home = home
        if not native:
            self._server_controller = "python"

        self._shm = None
        self._zmq_ctx = None
        self._zmq_sock = None
        self._connected = False
        self._server_proc = None

        # Rate limiting (mirrors FrankaController)
        self._update_freq = 50.0
        self._last_update_time = {}

        # Compatibility: no-op lock for code that uses `with controller.state_lock:`
        self.state_lock = threading.Lock()

    def start(self):
        """Start the server subprocess and connect.

        Spawns a 1kHz control loop in a child process, then connects to it
        via shared memory and ZMQ. The subprocess terminates automatically
        when this script exits.

        The robot must already be unlocked with FCI active (via
        ``aiofranka.unlock()`` or the Franka Desk web GUI). If not, the
        server subprocess will fail to start and a helpful error message
        is printed.

        Raises:
            RuntimeError: If the server fails to start.
        """
        from aiofranka.robot import require_pylibfranka
        from aiofranka.server import start_subprocess

        # Fail with installation instructions here, not in the server subprocess
        require_pylibfranka()

        _GREEN = "\033[32m"

        print(f"\n  {_BOLD}aiofranka{_RST} {_DIM}|{_RST} "
              f"starting server {_DIM}({self.robot_ip}){_RST}\n")

        # Spawn server subprocess (no homing — user script controls movement)
        try:
            self._server_proc = start_subprocess(self.robot_ip, controller_cls=self._server_controller)
        except RuntimeError as e:
            err = str(e).lower()
            if "unlock" in err or "fci" in err or "joint" in err or "not ready" in err:
                print(f"  {_RED}Robot is not ready.{_RST} Unlock first:\n")
                print(f"  Add to your script before {_BOLD}ctrl.start(){_RST}:")
                print(f"    {_BOLD}aiofranka.unlock(){_RST}\n")
                print(f"  Or run from the terminal right now:")
                print(f"    {_BOLD}$ aiofranka unlock{_RST}\n")
            raise

        # Register cleanup so the subprocess dies when the script exits.
        # Use stop() instead of _terminate_server() to send a graceful ZMQ
        # stop command before SIGTERM, giving the control loop time to exit
        # cleanly (avoids communication_constraints_violation reflex).
        atexit.register(self.stop)

        # Connect to the now-running server
        self._shm = StateBlock(self.robot_ip, create=False, track=False)

        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_sock.setsockopt(zmq.RCVTIMEO, 2000)
        self._zmq_sock.setsockopt(zmq.SNDTIMEO, 2000)
        self._zmq_sock.connect(zmq_endpoint_for_ip(self.robot_ip))

        resp = self._send({"cmd": "status"})
        if not resp.get("running"):
            raise RuntimeError("Server started but control loop is not running")

        self._connected = True
        print(f"  {_GREEN}Ready{_RST} {_DIM}(PID {self._server_proc.pid}){_RST}\n")

    def stop(self):
        """Stop the server subprocess and disconnect."""
        if self._connected:
            try:
                self._send({"cmd": "stop"})
            except Exception:
                pass
        self._disconnect()
        self._terminate_server()

    def _disconnect(self):
        if self._zmq_sock:
            self._zmq_sock.close()
            self._zmq_sock = None
        if self._zmq_ctx:
            self._zmq_ctx.term()
            self._zmq_ctx = None
        if self._shm:
            self._shm.close()
            self._shm = None
        self._connected = False

    def _terminate_server(self):
        """Terminate the server subprocess if it's still alive."""
        proc = self._server_proc
        if proc is None or not proc.is_alive():
            self._server_proc = None
            return
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        self._server_proc = None

    def _check_server_alive(self):
        """Raise if the server has died or errored."""
        if self._shm is None:
            raise RuntimeError("Not connected. Call start() first.")
        status = self._shm.read_status()
        if status == STATUS_ERROR:
            error_msg = self._shm.read_error()
            self._connected = False
            raise ServerDiedError(error_msg or "Unknown server error")

    @property
    def state(self) -> dict:
        """Read current robot state from shared memory (zero-copy, no IPC overhead)."""
        self._check_server_alive()
        return self._shm.read_state()

    @property
    def q_desired(self) -> np.ndarray:
        """Current desired joint positions (7,) from shared memory."""
        return self.state["q_desired"]

    @property
    def ee_desired(self) -> np.ndarray:
        """Current desired end-effector pose (4,4) from shared memory."""
        return self.state["ee_desired"]

    @property
    def torque(self) -> np.ndarray:
        """Current commanded torque (7,) from shared memory."""
        return self.state["torque"]

    @property
    def initial_qpos(self) -> np.ndarray:
        """Initial joint positions (7,) set at last switch() from shared memory."""
        return self.state["initial_qpos"]

    @property
    def initial_ee(self) -> np.ndarray:
        """Initial end-effector pose (4,4) set at last switch() from shared memory."""
        return self.state["initial_ee"]

    @property
    def running(self) -> bool:
        if not self._connected:
            return False
        try:
            resp = self._send({"cmd": "status"})
            return resp.get("running", False)
        except Exception:
            return False

    def switch(self, controller_type: str):
        """Switch control mode on the server ("impedance", "pid", "osc", "torque")."""
        self._send({"cmd": "switch", "type": controller_type})
        self._last_update_time.clear()

    def set_freq(self, freq: float):
        """Set update frequency for rate-limited set() calls (Hz)."""
        self._update_freq = freq
        self._send({"cmd": "set_freq", "freq": freq})

    def set(self, attr: str, value):
        """
        Rate-limited setter for controller attributes.

        Sends value to server and sleeps to maintain frequency from set_freq().

        Args:
            attr: "q_desired", "ee_desired", or "torque"
            value: numpy array
        """
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq
        value = np.asarray(value, dtype=np.float64)

        if attr not in self._last_update_time:
            self._last_update_time[attr] = current_time
            self._send_set(attr, value)
            time.sleep(dt)
            self._last_update_time[attr] = current_time + dt
            return

        target_time = self._last_update_time[attr] + dt
        self._send_set(attr, value)

        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        self._last_update_time[attr] = target_time

    def move(self, qpos=None):
        """
        Move to target joint position. Blocks until complete.

        Args:
            qpos: Target joint positions (7,). Default: home position.
        """
        if qpos is None:
            qpos = [0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853]

        qpos = np.asarray(qpos, dtype=np.float64)
        resp = self._send({"cmd": "move", "qpos": qpos.tobytes()})
        if not resp.get("ok"):
            raise RuntimeError(f"Move failed: {resp.get('error')}")

        move_id = resp["move_id"]
        while True:
            resp = self._send({"cmd": "move_status", "move_id": move_id})
            if resp.get("done"):
                if resp.get("error"):
                    raise RuntimeError(f"Move error: {resp['error']}")
                return
            time.sleep(0.05)

    def initialize(self):
        """Re-initialize controller state to current robot position."""
        self._send({"cmd": "initialize"})

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('robot_ip', 'state_lock'):
            super().__setattr__(name, value)
            return

        if name in _REMOTE_ATTRS and self.__dict__.get('_connected', False):
            self._send_set(name, value if not isinstance(value, np.ndarray) else value)
        else:
            super().__setattr__(name, value)

    # --- Internal ---

    def _send_set(self, attr: str, value):
        if isinstance(value, np.ndarray):
            msg = {
                "cmd": "set",
                "attr": attr,
                "value": value.astype(np.float64).tobytes(),
                "shape": list(value.shape),
            }
        else:
            msg = {"cmd": "set", "attr": attr, "value": value}
        self._send(msg)

    def _send(self, msg: dict) -> dict:
        if self._zmq_sock is None:
            raise RuntimeError("Not connected. Call start() first.")
        try:
            self._zmq_sock.send(msgpack.packb(msg, use_bin_type=True))
            raw = self._zmq_sock.recv()
            return msgpack.unpackb(raw, raw=False)
        except zmq.error.Again:
            # ZMQ timed out — check if the server died with an error
            server_error = None
            if self._shm is not None:
                try:
                    status = self._shm.read_status()
                    if status == STATUS_ERROR:
                        server_error = self._shm.read_error()
                except Exception:
                    pass

            # Fallback: if shm is gone but server process is dead, we know it crashed
            if not server_error and self._server_proc is not None:
                if not self._server_proc.is_alive():
                    server_error = "Server process died unexpectedly"

            if server_error:
                self._connected = False
                raise ServerDiedError(server_error) from None

            raise ConnectionError(
                "Server not responding (timeout). Check if server is running."
            ) from None

    def __del__(self):
        self._disconnect()
        self._terminate_server()


class FrankaRemoteControllerNative(FrankaRemoteController):
    """
    FrankaRemoteController whose server runs the 1 kHz control loop in C++.

    Same API as FrankaRemoteController. The server subprocess uses NativeServerController
    (aiofranka.server_native), so nothing in Python can delay a torque command. Since
    FrankaRemoteController's server does too by default, this class is kept for existing code.
    """

    _server_controller = "native"


def _load_last_ip() -> str:
    config_path = os.path.expanduser("~/.aiofranka/config.json")
    try:
        with open(config_path) as f:
            return json.load(f).get("last_ip", DEFAULT_IP)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_IP
