"""
aiofranka: Asyncio-based Franka Robot Control

A high-level Python library for controlling Franka Emika robots using asyncio.
Combines pylibfranka for real-time control with MuJoCo for kinematics/dynamics.

Main Components:
    RobotInterface: Low-level robot interface (real or simulation)
    NativeFrankaController: High-level asyncio controller with multiple modes, whose
        1 kHz loop runs in C++
    FrankaRemoteController: The same with a sync API, its loop in a server subprocess
    FrankaController: The legacy controller, with its 1 kHz loop in Python

Quick Example:
    >>> import asyncio
    >>> from aiofranka import RobotInterface, NativeFrankaController
    >>>
    >>> async def main():
    ...     robot = RobotInterface("172.16.0.2")
    ...     controller = NativeFrankaController(robot)
    ...     await controller.start()
    ...     await controller.move()  # Move to home
    ...     await controller.stop()
    >>>
    >>> asyncio.run(main())

For detailed documentation, see README.md and docs/
"""

from importlib.metadata import PackageNotFoundError, version as package_version

from aiofranka.controller import FrankaController
from aiofranka.native import NativeFrankaController, control_law
from aiofranka.robot import RobotInterface
from aiofranka.async_utils import asyncify, async_input, CudaInferenceThread, mpify
from aiofranka.remote import FrankaRemoteController, FrankaRemoteControllerNative, ServerDiedError
from aiofranka.remote_v2 import FrankaRemoteControllerV2
from aiofranka.server import start, stop, lock, unlock, set_configuration
from aiofranka.tools import Tool, active_tool, list_tools, load_tool, remove_tool, save_tool, unload_tool
from aiofranka.config import load_config

# Optional gripper support - only import if dependencies are available
try:
    from aiofranka.gripper import GripperController, RobotiqGripperInterface, create_gripper
    from aiofranka.gripper_remote import GripperRemoteController
    _HAS_ROBOTIQ = True
except ImportError:
    _HAS_ROBOTIQ = False

try:
    __version__ = package_version("aiofranka")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["RobotInterface", "FrankaController", "NativeFrankaController", "control_law", "FrankaRemoteController", "FrankaRemoteControllerNative", "FrankaRemoteControllerV2", "ServerDiedError", "asyncify", "async_input", "CudaInferenceThread", "mpify", "start", "stop", "lock", "unlock", "set_configuration", "Tool", "active_tool", "list_tools", "load_tool", "remove_tool", "save_tool", "unload_tool", "load_config"]

if _HAS_ROBOTIQ:
    __all__.extend(["GripperController", "GripperRemoteController", "RobotiqGripperInterface", "create_gripper"])
