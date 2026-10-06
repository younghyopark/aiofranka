"""
aiofranka's API: a Robot, the arm, and a Controller that drives it.

    >>> import aiofranka
    >>> robot = aiofranka.Robot("172.16.0.2")      # None: a simulated arm in MuJoCo
    >>> controller = aiofranka.Controller(robot)
    >>> controller.start()                          # takes torque control, starts the 1 kHz loop
    >>> controller.move()
    >>> controller.switch("osc")
    >>> controller.set_freq(50)
    >>> controller.set("ee_desired", target)
    >>> robot.state["ee"]
    >>> controller.stop()

Robot is what is true about the arm whoever drives it: its connection, its MuJoCo model with
the tool, and its state. Controller is how it is driven: the 1 kHz loop, which runs in C++ and
never waits for Python, and the modes, gains, targets, trajectories and recordings it runs.

Controller's methods are plain calls. It runs its event loop in a thread of its own, so your
code can block, sleep or compute while the loop holds the arm and move() plays its trajectory.
For asyncio code, NativeFrankaController is the same controller with awaitable methods, on a
RobotInterface.
"""

import asyncio
import atexit
import concurrent.futures
import threading
import weakref

import numpy as np

from aiofranka.controller import FrankaController
from aiofranka.native import NativeFrankaController
from aiofranka.robot import RobotInterface

HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]

# The controller's attributes that Controller reads and writes on its engine, besides the
# parameters of registered control laws.
_ATTRIBUTES = frozenset({
    "kp", "kd", "ki", "ee_kp", "ee_kd", "null_kp", "null_kd",
    "q_desired", "ee_desired", "torque", "initial_qpos", "initial_ee",
    "control_transform", "torque_limit", "torque_diff_limit", "clip", "type",
    "last_command", "error_integral", "law_memory", "arrival_tolerance",
    "realtime_priority", "realtime_cpu", "error_callback",
})

# Controllers whose loop runs, to stop when the interpreter exits.
_started = weakref.WeakSet()


@atexit.register
def _stop_started():
    for controller in list(_started):
        try:
            controller.stop()
        except Exception:
            pass


class Robot:
    """
    A Franka arm: its connection, its MuJoCo model with the tool, and its state.

    Connecting reads the arm, which needs FCI active (aiofranka.unlock()), and Desk's active
    end-effector profile, which is merged into the model. Nothing moves until a Controller
    starts; one Controller at a time drives the arm.

    Args:
        ip (str | None): The robot's IP address, or None for a simulated arm in MuJoCo,
            shown in a viewer
        read_tool (bool): Read Desk's active end-effector profile (robot.tool), which
            Controller.activate() checks configurations against

    Example:
        >>> robot = aiofranka.Robot("172.16.0.2")
        >>> robot.state["qpos"]
        >>> robot.tool
    """

    def __init__(self, ip=None, *, read_tool=True):
        self._setup(RobotInterface(ip, read_tool=read_tool))

    @classmethod
    def _wrap(cls, interface):
        """A Robot on a RobotInterface that is connected already."""
        robot = cls.__new__(cls)
        robot._setup(interface)
        return robot

    def _setup(self, interface):
        self._interface = interface
        self._controller = None  # the Controller that drives the arm, from start() to stop()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def ip(self):
        """The robot's IP address, or None in simulation."""
        return self._interface.ip

    @property
    def real(self):
        """Whether the arm is real; False in simulation."""
        return self._interface.real

    @property
    def model(self):
        """The arm's MuJoCo model, with the tool merged into its last link."""
        return self._interface.model

    @property
    def tool(self):
        """Desk's active end-effector profile when connecting (an aiofranka.Tool), or None."""
        return self._interface.tool

    @property
    def load(self):
        """The load set with set_load(): mass [kg], com [m] and inertia [kg m^2]."""
        return dict(self._interface.load)

    @property
    def payload(self):
        """
        The payload the robot compensates, which the model includes: Desk's end-effector
        profile and the load of set_load() (in simulation, the load). Its mass [kg], com in the
        flange frame [m] and inertia about the com [kg m^2].
        """
        return dict(self._interface.payload)

    @property
    def controller(self):
        """The Controller that drives the arm, or None."""
        return self._controller

    @property
    def state(self):
        """
        The arm's latest state, a dict of numpy arrays:

        - qpos, qvel: joint positions [rad] and velocities [rad/s] (7,)
        - ee: the flange pose in the base frame (4, 4)
        - jac: the flange's Jacobian, linear rows first (6, 7)
        - mm: the joint-space mass matrix (7, 7)
        - last_torque: the torque the robot was last commanded [Nm] (7,)

        While a controller drives the arm, it is what the 1 kHz loop read at its last cycle;
        otherwise the arm is read now.
        """
        self._check_open()
        engine = self._engine()
        if engine is None:
            return self._interface.state
        state = engine.state
        if state is not None:
            return state
        # The loop has not finished its first cycle: the arm as it was read last.
        interface = self._interface
        data = interface.data
        return {"qpos": np.array(data.qpos), "qvel": np.array(data.qvel), "ee": interface._ee(),
                "jac": interface._jacobian(), "mm": interface._mass_matrix(),
                "last_torque": np.array(data.ctrl)}

    @property
    def robot_state(self):
        """
        The libfranka state the arm sent last (pylibfranka.RobotState), with every number it
        reports: tau_J, O_F_ext_hat_K, control_command_success_rate, ... None in simulation.
        """
        self._check_open()
        interface = self._interface
        if not interface.real:
            return None
        if self._engine() is None:
            interface.sync_mj()  # read it now; while a loop runs, it keeps it current
        return interface.robot_state

    def set_load(self, mass, com=(0.0, 0.0, 0.0), inertia=(0.0, 0.0, 0.0)):
        """
        Set a load on the flange for this connection, e.g. a tool without a Desk profile: the
        robot compensates its gravity, and the model includes it. To keep a tool across
        connections, save it as a profile with aiofranka.save_tool() instead.

        Args:
            mass (float): Load mass [kg]; 0 removes the load
            com (array-like): Center of mass in the flange frame [m] (3,)
            inertia (array-like): Inertia about the center of mass in the flange frame
                [kg m^2], (3, 3) or its diagonal (3,)

        Raises:
            RuntimeError: If a controller drives the arm, as the robot rejects a load then
        """
        self._check_open()
        if self._controller is not None:
            raise RuntimeError("set_load() needs the arm without a controller: stop() it first")
        self._interface.set_load(mass, com, inertia)

    def close(self):
        """Stop the controller that drives the arm, if any, and disconnect."""
        if self._closed:
            return
        controller = self._controller
        if controller is not None:
            controller.stop()
        interface = self._interface
        if interface.real:
            interface.robot = None  # libfranka disconnects with its Robot
        else:
            viewer = getattr(interface, "viewer", None)
            if viewer is not None and hasattr(viewer, "close"):
                viewer.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        where = self.ip if self.real else "simulated"
        driven = f", driven by {self._controller!r}" if self._controller is not None else ""
        return f"<Robot {where}{driven}>"

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The robot is closed")

    def _attach(self, controller):
        """Let controller drive the arm; one controller at a time does."""
        self._check_open()
        with self._lock:
            holder = self._controller
            if holder is not None and holder is not controller:
                raise RuntimeError("Another Controller drives this robot: stop() it first")
            self._controller = controller

    def _detach(self, controller):
        with self._lock:
            if self._controller is controller:
                self._controller = None

    def _engine(self):
        """The native controller whose loop drives the arm now, or None."""
        controller = self._controller
        if controller is None or not controller.running:
            return None
        return controller._engine


class _Engine(NativeFrankaController):
    """Controller's NativeFrankaController: a loop error is kept for Controller to raise,
    instead of exiting the process from Controller's thread."""

    async def _run(self):
        try:
            error = await self._watch()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = str(e)
        if error:
            self.__dict__["_failure"] = error
            self._report(error)
            # The motion has ended; end torque control too, so that the arm can be read again
            try:
                self.robot.stop()
            except Exception:
                self.robot.torque_controller = None


class Controller:
    """
    Drives a Robot with the native 1 kHz control loop.

    The loop runs in C++, in a thread that never waits for Python. The control modes
    (impedance, pid, osc, torque and control laws), gains, targets and recordings are those
    of NativeFrankaController, whose documentation has the details; here they are plain
    calls. The controller runs its event loop in a thread of its own, so your code can block,
    sleep or compute while the loop holds the arm. A loop error, e.g. a reflex, stops the loop
    and is raised by the next call that commands the arm.

    Args:
        robot (Robot): The arm to drive

    Attributes:
        kp, kd, ki, ee_kp, ee_kd, null_kp, null_kd: Gains, as in NativeFrankaController
        q_desired, ee_desired, torque: Targets of the modes; set them with set() to hold a rate
        initial_qpos, initial_ee: The arm's pose at the last switch()

    Example:
        >>> robot = aiofranka.Robot("172.16.0.2")
        >>> controller = aiofranka.Controller(robot)
        >>> controller.start()
        >>> controller.move()
        >>> controller.switch("impedance")
        >>> controller.set_freq(50)
        >>> for target in targets:
        ...     controller.set("q_desired", target)
        >>> controller.stop()
    """

    def __init__(self, robot):
        if not isinstance(robot, Robot):
            raise TypeError("Controller drives an aiofranka.Robot; "
                            "for a RobotInterface, use NativeFrankaController")
        engine = _Engine(robot._interface)
        engine.__dict__["_failure"] = None
        attributes = self.__dict__
        attributes["robot"] = robot
        attributes["_engine"] = engine
        attributes["_events"] = None  # the event loop, in _thread, while started
        attributes["_thread"] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def running(self):
        """Whether the 1 kHz loop runs."""
        return self._engine.running

    @property
    def error(self):
        """The error that stopped the loop, e.g. a reflex, or None; start() clears it."""
        return self._engine.__dict__.get("_failure")

    def start(self):
        """
        Take the arm's torque control and start the 1 kHz loop, holding the arm where it is.
        Returns once the loop runs.

        Raises:
            RuntimeError: If another controller drives the robot
        """
        if self.running:
            return
        self._close()  # what a failed loop left
        robot = self.robot
        robot._attach(self)
        try:
            interface = robot._interface
            if interface.real:
                interface.sync_mj()  # the arm as it is now
            # Hold it there; NativeFrankaController.initialize() would first take the last
            # cycle of an earlier session.
            FrankaController.initialize(self._engine)
            self._engine.__dict__["_failure"] = None
            self._open()
            self._wait(self._engine.start())
            self._check()  # the loop may have stopped already
        except BaseException:
            self._close()
            raise
        _started.add(self)

    def stop(self):
        """Stop the 1 kHz loop and give the arm's torque control back. Does nothing if stopped."""
        _started.discard(self)
        self._close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ── Commanding ────────────────────────────────────────────────────────────

    def switch(self, controller_type):
        """
        Switch the control mode: "impedance", "pid", "osc", "torque", or a control law (see
        aiofranka.control_law). The targets move to the arm's current pose.
        """
        self._check()
        self._call(self._engine.switch, controller_type)

    def set_freq(self, freq):
        """Set the rate that set() holds [Hz], e.g. a policy's."""
        self._engine.set_freq(freq)

    def set(self, attr, value):
        """
        Set a target ("q_desired", "ee_desired" or "torque") or another attribute, then sleep
        to hold the rate of set_freq(), as a policy sends its actions: each call returns one
        period after the last one's target time, whatever the caller did in between.

        The controller's event loop sets it, as NativeFrankaController.set() does: it wakes up
        every millisecond, which keeps the period to about a cycle, where a sleep of the
        caller's could wake up milliseconds late.

        Args:
            attr (str): Attribute to set
            value: Its value, e.g. joint positions [rad] (7,) or a pose (4, 4)
        """
        self._check()
        self._require_running("set()")
        self._wait(self._engine.set(attr, value))

    def move(self, qpos=HOME):
        """
        Move to joint positions on a smooth, time-optimal trajectory, and hold them. Returns
        once every joint is within arrival_tolerance of them. Ctrl+C stops the motion, with
        the arm holding its last target.

        Args:
            qpos (array-like): Joint positions [rad] (7,), by default the home pose
        """
        self._check()
        self._require_running("move()")
        self._wait(self._engine.move(list(qpos)))

    def activate(self, config, check_tool=True, check_null_target=True):
        """
        Apply a controller configuration, a YAML file or a dict: its mode, gains, TCP and
        policy rate (see NativeFrankaController.activate()).

        Returns:
            dict: The configuration
        """
        self._check()
        return self._call(self._engine.activate, config, check_tool, check_null_target)

    def check_tool(self, config):
        """
        Check that the tool a configuration needs is Desk's active end-effector profile, as
        activate() does.

        Raises:
            RuntimeError: If another tool is active, or Desk could not be read
        """
        self._engine.check_tool(config)

    def set_tcp(self, transform):
        """Set the point the OSC controls: a translation [m] (3,) or a pose (4, 4) in the flange frame."""
        self._check()
        self._call(self._engine.set_tcp, transform)

    def register_law(self, law):
        """Compile a control law (aiofranka.control_law) and make it a controller type; returns its name."""
        return self._call(self._engine.register_law, law)

    def identify_payload(self, tool_length=0.2, tool_radius=0.1, floor=0.0, **kwargs):
        """
        Identify the mass and center of mass of the tool on the flange: moves the arm through
        poses around the current one, about 3 minutes (see aiofranka.payload.identify_payload()).
        A controller that is not running is started for it and stopped after. It only
        measures: save the result with aiofranka.save_tool().

        Returns:
            PayloadEstimate: The estimated tool, with standard errors
        """
        self._check()
        started = self.running
        if not started:
            self.robot._attach(self)
            self._open()
        try:
            return self._wait(self._engine.identify_payload(
                tool_length=tool_length, tool_radius=tool_radius, floor=floor, **kwargs))
        finally:
            if not started:
                self._close()

    # ── The loop ──────────────────────────────────────────────────────────────

    def record(self, fields=None, seconds=60.0, path=None):
        """
        Record every cycle of the loop, from its next one on (see NativeFrankaController.record()).

        Returns:
            Recording: The rows; stop() it, or use it in a with block
        """
        return self._call(self._engine.record, fields, seconds, path)

    def loop_stats(self, reset=False):
        """Timing of the loop since start, or since the last call with reset (NativeFrankaController.loop_stats())."""
        return self._engine.loop_stats(reset=reset)

    def step(self):
        """Run one cycle by hand while the loop is not running, e.g. to try a law in simulation."""
        self._engine.step()

    # ── Attributes ────────────────────────────────────────────────────────────

    def __getattr__(self, name):
        engine = self.__dict__.get("_engine")
        if engine is not None and (name in _ATTRIBUTES or name in engine._custom):
            return getattr(engine, name)
        if name == "state":
            raise AttributeError("Controller has no state: read the arm's, robot.state")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        engine = self.__dict__["_engine"]
        if name not in _ATTRIBUTES and name not in engine._custom:
            raise AttributeError(f"Controller has no attribute {name!r} to set; its attributes are "
                                 f"{', '.join(sorted(_ATTRIBUTES))} and control laws' parameters")
        self._check()
        self._assign(name, value)

    def __repr__(self):
        state = "running" if self.running else "stopped"
        return f"<Controller {self._engine.type}, {state}>"

    # ── Internals ─────────────────────────────────────────────────────────────

    def _assign(self, name, value):
        engine = self._engine
        if name in engine._layout or name in engine._custom:
            setattr(engine, name, value)  # the loop's memory, which it copies under a lock
        else:
            self._call(setattr, engine, name, value)

    def _check(self):
        error = self._engine.__dict__.get("_failure")
        if error:
            raise RuntimeError(f"The control loop stopped: {error}. Recover the robot "
                               "(aiofranka unlock) and start() the controller again")

    def _require_running(self, what):
        if not self.running:
            raise RuntimeError(f"{what} needs the loop: start() the controller first")

    def _open(self):
        """Start the event loop thread."""
        if self._events is not None:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_serve, args=(loop,), name="aiofranka-controller", daemon=True)
        self.__dict__.update(_events=loop, _thread=thread)
        thread.start()

    def _close(self):
        """Stop the loop and torque control, end the event loop thread and release the robot."""
        try:
            if self._events is not None:
                try:
                    self._wait(self._shutdown())
                finally:
                    loop, thread = self._events, self._thread
                    self.__dict__.update(_events=None, _thread=None)
                    loop.call_soon_threadsafe(loop.stop)
                    thread.join()
                    loop.close()
        finally:
            self.robot._detach(self)

    async def _shutdown(self):
        engine = self._engine
        engine.running = False  # asks the loop to stop
        task = engine.task
        if task is not None and not task.done():
            await task  # its watcher joins the loop and mirrors its last cycle
        try:
            engine.robot.stop()  # ends torque control
        except Exception:
            engine.robot.torque_controller = None
            if not engine.__dict__.get("_failure"):
                raise  # after a loop error, the motion has ended already

    def _call(self, function, *args):
        """Call function in the event loop thread while it runs, otherwise here."""
        if self._events is None or threading.current_thread() is self._thread:
            return function(*args)

        async def call():
            return function(*args)

        return self._wait(call())

    def _wait(self, coroutine):
        """Run coroutine in the event loop thread and return its result."""
        if threading.current_thread() is self._thread:
            coroutine.close()
            raise RuntimeError("This Controller method cannot run in the controller's own thread, "
                               "e.g. from error_callback; call it from your code")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._events)
        try:
            while True:
                try:
                    # Waking up every 50 ms takes Ctrl+C, which may reach another thread
                    return future.result(timeout=0.05)
                except concurrent.futures.TimeoutError:
                    if future.done():
                        return future.result()  # it raised a TimeoutError of its own
        except BaseException:
            future.cancel()  # e.g. Ctrl+C in move(): the arm holds its last target
            raise


def _serve(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()
