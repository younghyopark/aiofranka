"""
V2 remote controller client — uses the RT-threaded server for minimal jitter.

Drop-in replacement for FrankaRemoteController. The only difference is that
the server subprocess uses ServerControllerV2 (dedicated RT thread) instead
of the asyncio-based ServerController. Also includes client-side jitter
warnings read from shared memory.

Legacy: its loop runs in Python. FrankaRemoteController's server runs the loop
in C++ (NativeServerController), which nothing in Python can delay.
"""

import logging
import time

from aiofranka.ipc import StateBlock, zmq_endpoint_for_ip
from aiofranka.remote import FrankaRemoteController, _BOLD, _DIM, _RED, _YELLOW, _RST

logger = logging.getLogger("aiofranka.remote")


class FrankaRemoteControllerV2(FrankaRemoteController):
    """FrankaRemoteController that launches a v2 (RT-threaded) server.

    Identical API to FrankaRemoteController. The only change is the server
    subprocess uses a dedicated RT thread for the 1kHz control loop instead
    of asyncio, reducing jitter on PREEMPT_RT systems.

    Includes client-side jitter warnings (v2 server writes stats to shm).

    Legacy: the v2 loop runs in Python. FrankaRemoteController's server runs it in C++.
    """

    def __init__(self, robot_ip=None, *, home=True):
        super().__init__(robot_ip, home=home)
        # Jitter monitoring (tracks cumulative counts from server)
        self._last_jitter_warn = 0
        self._last_jitter_error = 0
        self._jitter_check_interval = 1.0
        self._last_jitter_check = 0.0

    def start(self):
        """Start the v2 server subprocess and connect."""
        from aiofranka.robot import require_pylibfranka
        from aiofranka.server_v2 import start_subprocess_v2

        # Fail with installation instructions here, not in the server subprocess
        require_pylibfranka()

        _GREEN = "\033[32m"

        print(f"\n  {_BOLD}aiofranka v2{_RST} {_DIM}|{_RST} "
              f"starting RT server {_DIM}({self.robot_ip}){_RST}\n")

        try:
            self._server_proc = start_subprocess_v2(self.robot_ip)
        except RuntimeError as e:
            err = str(e).lower()
            if "unlock" in err or "fci" in err or "joint" in err or "not ready" in err:
                print(f"  {_RED}Robot is not ready.{_RST} Unlock first:\n")
                print(f"  Add to your script before {_BOLD}ctrl.start(){_RST}:")
                print(f"    {_BOLD}aiofranka.unlock(){_RST}\n")
                print(f"  Or run from the terminal right now:")
                print(f"    {_BOLD}$ aiofranka unlock{_RST}\n")
            raise

        import atexit
        atexit.register(self.stop)

        self._shm = StateBlock(self.robot_ip, create=False, track=False)

        import zmq
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_sock.setsockopt(zmq.RCVTIMEO, 2000)
        self._zmq_sock.setsockopt(zmq.SNDTIMEO, 2000)
        self._zmq_sock.connect(zmq_endpoint_for_ip(self.robot_ip))

        resp = self._send({"cmd": "status"})
        if not resp.get("running"):
            raise RuntimeError("Server started but control loop is not running")

        self._connected = True
        print(f"  {_GREEN}Ready{_RST} {_DIM}(RT thread, PID {self._server_proc.pid}){_RST}\n")

    @property
    def state(self):
        """Read state from shared memory, with jitter warnings."""
        self._check_server_alive()
        self._check_jitter()
        return self._shm.read_state()

    def _check_jitter(self):
        """Print client-side warning when server reports new jitter events."""
        now = time.monotonic()
        if now - self._last_jitter_check < self._jitter_check_interval:
            return
        self._last_jitter_check = now

        try:
            max_dt_ms, warn_count, error_count = self._shm.read_jitter_stats()
        except Exception:
            return

        new_warns = warn_count - self._last_jitter_warn
        new_errors = error_count - self._last_jitter_error

        if new_errors > 0:
            print(
                f"  {_RED}[Jitter]{_RST} 1kHz control loop interrupted: "
                f"{new_errors} error(s), max dt={max_dt_ms:.1f}ms"
            )
        elif new_warns > 0:
            print(
                f"  {_YELLOW}[Jitter]{_RST} 1kHz control loop unstable: "
                f"{new_warns} warning(s), max dt={max_dt_ms:.1f}ms"
            )

        self._last_jitter_warn = warn_count
        self._last_jitter_error = error_count
