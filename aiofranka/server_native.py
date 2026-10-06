"""
The aiofranka server with the native control loop.

NativeServerController is ServerController on NativeFrankaController: the server process
runs the 1 kHz loop in C++ and keeps writing the robot state to shared memory, so clients
(FrankaRemoteController) work unchanged. The server runs it by default:

    $ aiofranka start-server

as do aiofranka.start() and FrankaRemoteController. aiofranka start-server --python,
aiofranka.start(native=False) and FrankaRemoteController(native=False) run the legacy
ServerController, with the loop in Python.
"""

import asyncio
import logging
import os
import sys

from aiofranka.ipc import StateBlock
from aiofranka.native import NativeFrankaController
from aiofranka.robot import RobotInterface

logger = logging.getLogger("aiofranka.server")


class NativeServerController(NativeFrankaController):
    """
    The server's controller with the native loop, a drop-in for ServerController.

    Writes the state to shared memory about every millisecond, from the event loop. On an
    error it reports to shared memory and returns, without exiting, so the server can retry.
    """

    def __init__(self, robot: RobotInterface, shm_block: StateBlock):
        super().__init__(robot)
        self._shm = shm_block
        self._last_error = None

    async def start(self):
        """Start torque control in a thread with a timeout, then the native loop."""
        loop = asyncio.get_event_loop()
        logger.info("Starting torque control (native loop)...")
        try:
            # pylibfranka's start_torque_control() can hang after a reflex error.
            await asyncio.wait_for(loop.run_in_executor(None, self.robot.start), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError("robot.start() timed out (10s)")
        if self.task is None or self.task.done():
            if sys.platform == "darwin":
                self._launch()
            else:
                # Unless realtime_cpu says otherwise, the last core at SCHED_FIFO priority 80,
                # like ServerController.
                cpu = self.realtime_cpu if self.realtime_cpu is not None else (os.cpu_count() or 1) - 1
                self._launch(cpu=cpu, fifo_priority=80)
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(1)
        return self.task

    def _tick(self):
        state = self.state
        if state is None:
            return
        self._shm.write_state({
            **state,
            "q_desired": self.q_desired,
            "ee_desired": self.ee_desired,
            "torque": self.torque,
            "initial_qpos": self.initial_qpos,
            "initial_ee": self.initial_ee,
        })
        self._shm.write_ctrl_type(self.type)
        stats = self._loop.stats()
        self._shm.write_jitter_stats(stats["max_all"] * 1000.0, stats["warn"], stats["error"])

    async def _run(self):
        """Watch the native loop; report an error without exiting, for the server to retry."""
        try:
            error = await self._watch()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = str(e)
        if error:
            self._last_error = error
            logger.error(f"Control loop error: {error}")
            self._shm.write_error(error)
            if self.error_callback is not None:
                try:
                    self.error_callback(error)
                except Exception as cb_err:
                    logger.error(f"Error in error_callback: {cb_err}")
