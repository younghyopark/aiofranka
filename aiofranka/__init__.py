"""
aiofranka: Asyncio-based Franka Robot Control

A high-level Python library for controlling Franka Emika robots using asyncio.
Combines pylibfranka for real-time control with MuJoCo for kinematics/dynamics.

Main Components:
    Robot: The arm, real or simulated: its connection, its model and its state
    Controller: Drives a Robot with the 1 kHz loop in C++, with plain calls
    NativeFrankaController: The same controller with awaitable methods, for asyncio code
    FrankaRemoteController: Legacy server mode, the loop in a subprocess
    FrankaController: The legacy controller, with its 1 kHz loop in Python

Quick Example:
    >>> import aiofranka
    >>>
    >>> robot = aiofranka.Robot("172.16.0.2")   # None: MuJoCo
    >>> with aiofranka.Controller(robot) as controller:
    ...     controller.move()  # Move to home
    ...     print(robot.state["qpos"])

For detailed documentation, see README.md and docs/
"""

from importlib.metadata import PackageNotFoundError, version as package_version

from aiofranka.controller import FrankaController
from aiofranka.franka import Controller, Robot
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

__all__ = ["Robot", "Controller", "RobotInterface", "FrankaController", "NativeFrankaController", "control_law", "FrankaRemoteController", "FrankaRemoteControllerNative", "FrankaRemoteControllerV2", "ServerDiedError", "asyncify", "async_input", "CudaInferenceThread", "mpify", "start", "stop", "lock", "unlock", "set_configuration", "Tool", "active_tool", "list_tools", "load_tool", "remove_tool", "save_tool", "unload_tool", "load_config"]

if _HAS_ROBOTIQ:
    __all__.extend(["GripperController", "GripperRemoteController", "RobotiqGripperInterface", "create_gripper"])
