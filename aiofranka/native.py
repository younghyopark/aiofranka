"""
The native control loop: NativeFrankaController, a drop-in for FrankaController.

FrankaController runs its 1 kHz control loop on the asyncio event loop, in Python. Anything
else that runs there delays the next torque command: a planner, loading a model, garbage
collection, the server's command thread holding the GIL. When a command is late, the robot
stops with communication_constraints_violation. NativeFrankaController runs the same loop
in C++, in a thread that never waits for Python, and has FrankaController's constructor,
methods and attributes, so porting code means swapping the class:

    >>> robot = RobotInterface("172.16.0.2")
    >>> controller = NativeFrankaController(robot)  # instead of FrankaController(robot)
    >>> await controller.start()
    >>> controller.switch("osc")
    >>> await controller.set("ee_desired", target)

Its impedance, pid, osc and torque laws are ports of FrankaController's and send the same
torques, to rounding. New laws are written in Python and compiled with Numba, see
control_law().

What changes:
    - The loop runs in C++, so a subclass's step() would never run: NativeFrankaController
      refuses one. Write the step as a control law instead, and log every cycle with
      record().
    - Attributes the loop reads (kp, q_desired, ee_desired, ...) are views of its memory.
      Assigning one copies the value in, and the loop takes it whole at its next cycle.
    - While the loop runs, robot.data and robot.robot_state follow it from the event loop,
      about every millisecond. In simulation the loop owns the simulated arm.
"""

import asyncio
import copy
import ctypes
import glob
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path

import mujoco
import numpy as np

from aiofranka.controller import FrankaController

logger = logging.getLogger(__name__)

# Controller types whose torque FrankaController keeps in .torque, which torque mode then sends.
_KEEPS_TORQUE = ("impedance", "pid")

# What NativeFrankaController.record() records when it is not given fields.
RECORD_FIELDS = ("cycle", "time", "wall_time", "busy", "q", "dq", "tcp", "q_desired", "ee_desired",
                 "tau", "tau_J_d", "tau_J", "control_command_success_rate")

_BUILD_HELP = """aiofranka's native control loop, the extension aiofranka._native, is not built.

Build it in the aiofranka checkout, which needs a C++17 compiler:

    pip install "pybind11>=3.1,<3.2"
    pip install --no-build-isolation -e .
"""

_native = None


def _promote_libfranka(pylibfranka):
    """Make the libfranka that pylibfranka loaded global, so aiofranka._native binds to it."""
    package = os.path.dirname(pylibfranka.__file__)
    candidates = []
    for directory in (package, package + ".libs"):
        for pattern in ("libfranka*.dylib", "libfranka*.so*"):
            candidates += sorted(glob.glob(os.path.join(directory, pattern)))
    for path in candidates:
        try:
            # RTLD_NOLOAD: only the copy that is loaded already, never a second one.
            ctypes.CDLL(path, mode=os.RTLD_GLOBAL | os.RTLD_NOLOAD)
            return path
        except OSError:
            continue
    raise ImportError(f"Could not find the libfranka that pylibfranka loaded (looked in {package})")


def load_native():
    """
    Import the native control loop, aiofranka._native.

    It uses pylibfranka's libfranka, which it must have been built for.

    Returns:
        module: aiofranka._native

    Raises:
        ImportError: If the extension is not built, or was built for another libfranka
    """
    global _native
    if _native is not None:
        return _native
    from aiofranka.robot import require_pylibfranka

    pylibfranka = require_pylibfranka()
    import pylibfranka._pylibfranka  # noqa: F401  (loads libfranka)

    _promote_libfranka(pylibfranka)
    try:
        from aiofranka import _native as native
    except ImportError as error:
        raise ImportError(f"{_BUILD_HELP}\n({error})") from error
    installed = getattr(pylibfranka, "__version__", "")
    if installed.split(".")[:2] != native.LIBFRANKA_VERSION.split(".")[:2]:
        raise ImportError(
            f"aiofranka's native control loop was built for libfranka {native.LIBFRANKA_VERSION}, "
            f"but pylibfranka {installed} is installed. Update the headers in src/native/include "
            "to that version and rebuild aiofranka.")
    _native = native
    return native


def _mujoco_functions():
    """The addresses of the MuJoCo functions the loop calls, in the libmujoco mujoco loaded."""
    directory = os.path.dirname(mujoco.__file__)
    for pattern in ("libmujoco*.dylib", "libmujoco.so*", "mujoco*.dll"):
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            try:
                library = ctypes.CDLL(path, mode=os.RTLD_NOLOAD)
            except OSError:
                continue
            names = ("mj_fwdPosition", "mj_step", "mj_jacSite", "mj_fullM")
            return {name: ctypes.cast(getattr(library, name), ctypes.c_void_p).value for name in names}
    raise ImportError(f"Could not find the libmujoco that mujoco loaded (looked in {directory})")


def _dtype(layout, itemsize, extra=()):
    """A numpy structured dtype of a C++ struct, from the layout the module exports."""
    names, formats, offsets = [], [], []
    for name, offset, shape, kind in tuple(layout) + tuple(extra):
        names.append(name)
        offsets.append(offset)
        formats.append((np.dtype(kind), shape) if shape else np.dtype(kind))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": itemsize})


def _shapes(fields):
    """{name: shape} from {name: shape or size}."""
    out = {}
    for name, shape in (fields or {}).items():
        if not name.isidentifier():
            raise ValueError(f"{name!r} is not a valid field name")
        out[name] = (shape,) if isinstance(shape, int) else tuple(shape)
    return out


class ControlLaw:
    """
    A control law for NativeFrankaController, made with control_law().

    Attributes:
        name (str): The controller type that runs it, for switch()
        function: The Python function
        params (dict): Shapes of its parameters, by name
        memory (dict): Shapes of its memory, by name
    """

    def __init__(self, function, params=None, memory=None, name=None):
        self.function = function
        self.name = name or function.__name__
        self.params = _shapes(params)
        self.memory = _shapes(memory)
        self.__doc__ = function.__doc__

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return f"<ControlLaw {self.name}>"


def control_law(function=None, *, params=None, memory=None, name=None):
    """
    Make a Python function a control law of NativeFrankaController.

    The native loop calls it at 1 kHz, compiled with Numba, so it runs without Python: write
    it with numpy and math only (Numba's nopython mode). It gets four arguments:

    - s: what the loop read at this cycle, FrankaController.state's fields: qpos, qvel,
      ee (flange pose, 4x4), jac (6x7, linear rows first), mm (7x7), last_torque,
      and cycle, time [s] and dt (1e-3)
    - p: the controller's attributes: kp, kd, ki, ee_kp, ee_kd, null_kp, null_kd,
      q_desired, ee_desired, torque, initial_qpos, control_transform, torque_limit,
      torque_diff_limit, plus the law's own params
    - m: the law's memory, zeroed when switch() selects the law; write to it
    - tau: the 7 torques to fill [Nm]

    Return nothing, or a nonzero integer to stop the loop with an error. With clip (the
    default), the loop then limits the torque rate and clips the torques, as the impedance
    law does.

    Each of params becomes a controller attribute (zeros at first), which set() and
    activate() can change like kp. Laws that name the same parameter share it.

    Args:
        params (dict): Shapes of the law's parameters, by name, e.g. {"stiffness": 7}
        memory (dict): Shapes of what the law keeps between cycles, e.g. {"integral": 7}
        name (str): The controller type for switch() (default: the function's name)

    Example:
        >>> @control_law(params={"stiffness": 7, "damping": 7}, memory={"integral": 7})
        ... def pi_damping(s, p, m, tau):
        ...     error = p.q_desired - s.qpos
        ...     m.integral[:] += error * s.dt
        ...     tau[:] = p.stiffness * error + 2.0 * m.integral - p.damping * s.qvel
        >>> controller.switch(pi_damping)  # compiles it, a few seconds the first time
        >>> controller.stiffness = np.full(7, 60.0)
        >>> await controller.set("q_desired", target)
    """
    def decorate(f):
        return ControlLaw(f, params=params, memory=memory, name=name)
    return decorate if function is None else decorate(function)


def _compile(law, native, params_dtype, memory_dtype):
    """Compile a law with Numba into the C function the loop calls."""
    try:
        import numba
        from numba import carray, types
    except ImportError as error:
        raise ImportError("Custom control laws need Numba: pip install numba") from error

    function = numba.njit(law.function, error_model="numpy")
    state = numba.from_dtype(_dtype(native.STATE_LAYOUT, native.STATE_SIZE))
    params = numba.from_dtype(params_dtype)
    memory = numba.from_dtype(memory_dtype)
    signature = types.int32(types.CPointer(state), types.CPointer(params),
                            types.CPointer(memory), types.CPointer(types.float64))
    # What the law returns decides the wrapper: Numba types every branch of one.
    function.compile((state, params, memory, types.float64[::1]))
    returns = function.nopython_signatures[-1].return_type

    if returns == types.none:
        def call(s, p, m, tau):
            function(carray(s, 1)[0], carray(p, 1)[0], carray(m, 1)[0], carray(tau, (7,)))
            return 0
    else:
        def call(s, p, m, tau):
            code = function(carray(s, 1)[0], carray(p, 1)[0], carray(m, 1)[0], carray(tau, (7,)))
            if code is None:
                return 0
            return code

    return numba.cfunc(signature, nopython=True, error_model="numpy")(call)


class _Param:
    """A controller attribute that the native loop reads: a view of its staging memory."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, controller, owner=None):
        if controller is None:
            return self
        return controller._views[self.name]

    def __set__(self, controller, value):
        controller._assign(self.name, value)


class Recording:
    """
    Every cycle of the native loop, from NativeFrankaController.record().

    The loop writes a row per cycle into a buffer in C++, which the controller moves to
    Python about every 10 ms. stop() ends the recording; using it in a with block stops it
    at the end of the block.

    Attributes:
        fields (tuple): The recorded fields
        path (Path): Where stop() saves the rows (.npz), or None
    """

    def __init__(self, loop, recorder, fields, dtype, path):
        self.fields = tuple(fields)
        self.path = None if path is None else Path(path)
        self._loop = loop
        self._recorder = recorder  # until stop(), which keeps its counts
        self._counts = None
        self._dtype = dtype
        self._chunks = []
        self._lock = threading.Lock()
        self._active = True

    @property
    def recording(self):
        """Whether the loop still records into it."""
        return self._active

    @property
    def rows(self):
        """Rows the loop recorded so far."""
        recorder = self._recorder
        return self._counts[0] if recorder is None else recorder.rows

    @property
    def dropped(self):
        """Rows the loop dropped because Python had not taken the earlier ones in time."""
        recorder = self._recorder
        return self._counts[1] if recorder is None else recorder.dropped

    def _drain(self):
        with self._lock:
            if self._recorder is None:
                return
            chunk = self._recorder.drain()
            if chunk.size:
                self._chunks.append(chunk)

    def data(self):
        """
        The rows so far.

        Returns:
            dict: {field: numpy array with one row per cycle, oldest first}
        """
        self._drain()
        with self._lock:
            if len(self._chunks) > 1:
                self._chunks = [np.concatenate(self._chunks)]
            raw = self._chunks[0] if self._chunks else np.zeros(0, np.uint8)
        rows = raw.view(self._dtype)
        return {name: np.array(rows[name]) for name in self.fields}

    def save(self, path=None):
        """
        Save the rows so far to an .npz file, with one array per field.

        Args:
            path (str or Path): Where (default: the recording's path)

        Returns:
            Path: The file
        """
        path = Path(path if path is not None else self.path)
        data = self.data()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole or not at all.
        partial = path.with_name(f".{path.stem}.partial.npz")
        np.savez(partial, **data)
        partial.replace(path)
        return path

    def stop(self):
        """
        Stop recording, and save to the recording's path if it has one.

        Returns:
            dict: data()
        """
        if self._active:
            self._active = False
            self._loop.set_recorder(None)
            self._drain()
            with self._lock:
                # The rows are in Python now: let the buffer in C++ go.
                recorder, self._recorder = self._recorder, None
                self._counts = (recorder.rows, recorder.dropped)
            if self.path is not None:
                self.save()
                if self.dropped:
                    logger.warning(f"The recording in {self.path} lacks {self.dropped} rows the loop dropped")
        return self.data()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def __repr__(self):
        state = "recording" if self._active else "stopped"
        return f"<Recording {state}, {self.rows} rows of {', '.join(self.fields)}>"


class NativeFrankaController(FrankaController):
    """
    FrankaController with its 1 kHz control loop in C++.

    Same constructor, methods and attributes as FrankaController; see aiofranka.native for
    what changes. Requires the extension aiofranka._native.

    Example:
        >>> robot = RobotInterface("172.16.0.2")
        >>> controller = NativeFrankaController(robot)
        >>> await controller.start()
        >>> await controller.move()
        >>> controller.switch("impedance")
        >>> controller.set_freq(50)
        >>> await controller.set("q_desired", target)
        >>> await controller.stop()
    """

    kp = _Param()
    kd = _Param()
    ki = _Param()
    ee_kp = _Param()
    ee_kd = _Param()
    null_kp = _Param()
    null_kd = _Param()
    q_desired = _Param()
    ee_desired = _Param()
    initial_qpos = _Param()
    control_transform = _Param()
    torque_limit = _Param()

    # SCHED_FIFO priority of the loop's thread on Linux, which keeps other processes from
    # delaying it when every core is busy; 0 leaves it at normal priority. It needs an rtprio
    # limit (ulimit -r) of at least this, as a PREEMPT_RT setup for libfranka has.
    realtime_priority = 80

    # CPU that the loop's thread runs on, on Linux. None leaves it on the CPUs of the thread
    # that calls start(), whose scheduling it would share: give it a CPU of its own that no
    # other SCHED_FIFO thread uses.
    realtime_cpu = None

    # Seconds between syncs of the MuJoCo viewer in simulation, and between moves of the
    # recorded rows to Python.
    _VIEWER_PERIOD = 1 / 60
    _DRAIN_PERIOD = 0.01

    def __init__(self, robot):
        if type(self).step is not NativeFrankaController.step:
            raise TypeError(
                f"{type(self).__name__} overrides step(), which NativeFrankaController never calls: "
                "its 1 kHz loop runs in C++. Write the step as a control law "
                "(aiofranka.native.control_law), or subclass FrankaController.")
        native = load_native()
        loop = native.Loop()
        layout = {name: (offset, shape) for name, offset, shape, _ in native.PARAMS_LAYOUT}
        records = loop.params_buffer().view(_dtype(native.PARAMS_LAYOUT, native.PARAMS_SIZE))
        attributes = self.__dict__
        attributes["_native"] = native
        attributes["_loop"] = loop
        attributes["_layout"] = layout
        attributes["_records"] = records
        attributes["_views"] = {name: records[name][0] for name, (_, shape) in layout.items() if shape}
        attributes["_custom"] = {}        # custom law parameters: name -> (offset, shape)
        attributes["_custom_used"] = 0    # doubles of Params.custom given out
        attributes["_laws"] = {}          # name -> (ControlLaw, index, cfunc, memory dtype)
        attributes["_world_lock"] = threading.Lock()
        attributes["_synced_cycle"] = -1
        attributes["_models"] = {}        # model copies the loop may use, by epoch
        attributes["_configured"] = False
        attributes["_torque_diff_limit"] = 990.0
        attributes["_recording"] = None
        super().__init__(robot)

    # ── Attributes ────────────────────────────────────────────────────────────

    def _assign(self, name, value):
        if name in self._layout:
            offset, shape = self._layout[name]
        else:
            offset, shape = self._custom[name]
        array = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
        self._loop.assign(offset, np.ascontiguousarray(array).reshape(-1))

    def __getattr__(self, name):
        custom = self.__dict__.get("_custom")
        if custom and name in custom:
            return self._views[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        custom = self.__dict__.get("_custom")
        if custom and name in custom:
            self._assign(name, value)
        else:
            super().__setattr__(name, value)

    @property
    def type(self):
        """Controller type: "impedance", "pid", "osc", "torque", or a custom law's name."""
        return self.__dict__.get("_type", "impedance")

    @type.setter
    def type(self, name):
        if isinstance(name, ControlLaw):
            name = self.register_law(name)
        modes = self._native.MODES
        if name in modes and name != "custom":
            mode, index = modes[name], -1
        elif name in self._laws:
            mode, index = modes["custom"], self._laws[name][1]
        else:
            raise ValueError(f"Unknown controller type: {name}")
        previous = self.__dict__.get("_type")
        if previous in _KEEPS_TORQUE and name not in _KEEPS_TORQUE:
            # FrankaController leaves the impedance or PID torque in .torque, so torque mode
            # starts from it.
            snapshot = self._loop.snapshot()
            if snapshot is not None:
                self._assign("torque", snapshot["torque"])
        if index >= 0 and name != previous:
            self._loop.reset_memory()
        self._loop.assign_mode(mode, index)
        self.__dict__["_type"] = name

    @property
    def clip(self):
        """Whether the loop limits the torque rate and clips the torques."""
        return bool(self._records["clip"][0])

    @clip.setter
    def clip(self, value):
        self._loop.assign_int(self._layout["clip"][0], int(bool(value)))

    @property
    def torque_diff_limit(self):
        """Max torque rate [Nm/s], a number or one per joint."""
        return self.__dict__["_torque_diff_limit"]

    @torque_diff_limit.setter
    def torque_diff_limit(self, value):
        self._assign("torque_diff_limit", value)
        self.__dict__["_torque_diff_limit"] = value

    @property
    def torque(self):
        """
        The torque command of torque mode [Nm] (7,). In impedance and PID mode, the torque
        that the last cycle computed, as in FrankaController.
        """
        if self.type in _KEEPS_TORQUE:
            snapshot = self._loop.snapshot()
            if snapshot is not None:
                return snapshot["torque"]
        return self._views["torque"]

    @torque.setter
    def torque(self, value):
        self._assign("torque", value)

    @property
    def running(self):
        """Whether the native loop runs. Setting False stops it."""
        return self._loop.running()

    @running.setter
    def running(self, value):
        if not value:
            self._loop.request_stop()

    @property
    def state(self):
        """What the last cycle read: qpos, qvel, ee, jac, mm and last_torque."""
        state = self._loop.state_dict()
        return self.__dict__.get("_state") if state is None else state

    @state.setter
    def state(self, value):
        self.__dict__["_state"] = value

    @property
    def last_command(self):
        """The torque the last cycle sent, after the rate limit and clip [Nm] (7,)."""
        snapshot = self._loop.snapshot()
        return self.__dict__.get("_last_command") if snapshot is None else snapshot["last_command"]

    @last_command.setter
    def last_command(self, value):
        self.__dict__["_last_command"] = np.asarray(value, dtype=float)

    @property
    def error_integral(self):
        """The PID controller's integral of the joint error (7,). Setting it resets it."""
        snapshot = self._loop.snapshot()
        if snapshot is None:
            return self.__dict__.get("_error_integral", np.zeros(7))
        return snapshot["error_integral"]

    @error_integral.setter
    def error_integral(self, value):
        value = np.broadcast_to(np.asarray(value, dtype=np.float64), (7,)).copy()
        self.__dict__["_error_integral"] = value
        self._loop.reset_integral(value)

    @property
    def law_memory(self):
        """The memory of the running custom law, a copy with its fields, or None."""
        entry = self._laws.get(self.type)
        snapshot = self._loop.snapshot()
        if entry is None or snapshot is None:
            return None
        memory = np.zeros(1, dtype=entry[3])
        raw = memory.view(np.float64)
        raw[:len(snapshot["memory"])] = snapshot["memory"]
        return memory[0]

    # ── Custom laws ───────────────────────────────────────────────────────────

    def register_law(self, law):
        """
        Compile a control law and make it a controller type.

        switch() then takes its name; switch(law) registers it too. Compiling takes a few
        seconds the first time, during which the running loop carries on.

        Args:
            law (ControlLaw): A function decorated with control_law()

        Returns:
            str: The law's name
        """
        if not isinstance(law, ControlLaw):
            raise TypeError("register_law() takes a function decorated with aiofranka.native.control_law")
        if law.name in self._laws:
            if self._laws[law.name][0] is law:
                return law.name
            raise ValueError(f"A control law named {law.name!r} is registered already")
        native = self._native
        if law.name in native.MODES:
            raise ValueError(f"{law.name!r} is a built-in controller type")
        # Parameters: shared by name, in Params.custom.
        custom = dict(self._custom)
        used = self._custom_used
        base = self._layout["custom"][0]
        for name, shape in law.params.items():
            if name in self._layout or name in self.__dict__ or hasattr(type(self), name):
                raise ValueError(f"{law.name}: parameter {name!r} is a controller attribute already")
            if name in custom:
                if custom[name][1] != shape:
                    raise ValueError(f"{law.name}: parameter {name!r} has shape {custom[name][1]} in another law")
                continue
            size = int(np.prod(shape))
            if used + size > native.CUSTOM_PARAMS:
                raise ValueError(f"Custom laws have {native.CUSTOM_PARAMS} doubles of parameters in all")
            custom[name] = (base + 8 * used, shape)
            used += size
        params_dtype = _dtype(
            [f for f in native.PARAMS_LAYOUT if f[0] != "custom"], native.PARAMS_SIZE,
            [(name, offset, shape, "f8") for name, (offset, shape) in custom.items()])
        memory_layout, offset = [], 0
        for name, shape in law.memory.items():
            memory_layout.append((name, offset, shape, "f8"))
            offset += 8 * int(np.prod(shape))
        if offset > 8 * native.MEMORY:
            raise ValueError(f"A law's memory holds at most {native.MEMORY} doubles")
        memory_dtype = _dtype([], 8 * native.MEMORY, memory_layout)

        cfunc = _compile(law, native, params_dtype, memory_dtype)
        index = self._loop.register_law(law.name, cfunc.address, offset // 8)

        records = self._loop.params_buffer().view(params_dtype)
        views = dict(self._views)
        for name in custom:
            views[name] = records[name][0]
        self.__dict__.update(_custom=custom, _custom_used=used, _views=views)
        self._laws[law.name] = (law, index, cfunc, memory_dtype)
        return law.name

    def switch(self, controller_type: "str | ControlLaw"):
        """
        FrankaController.switch(), which also takes a control law (see control_law()).

        Like the PID integral, a custom law's memory starts from zero.
        """
        if isinstance(controller_type, ControlLaw):
            controller_type = self.register_law(controller_type)
        super().switch(controller_type)
        if controller_type in self._laws:
            self._loop.reset_memory()

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, fields=None, seconds=60.0, path=None):
        """
        Record every cycle of the native loop, from its next one on.

        Each cycle, after sending the torques, the loop writes the fields into a buffer in
        C++ without waiting for Python, and the controller moves them to Python about every
        10 ms. If the event loop is blocked for longer than `seconds`, the loop drops rows
        and Recording.dropped counts them.

        Fields, by name:

        - of the cycle: cycle; time, the robot's [s] (in simulation, the simulation's);
          wall_time, the host's time.time() when the robot state arrived [s]; busy, from
          then to the torque command [s]; q, dq, ee (flange pose), tcp (ee @
          control_transform), jac, mm; last_torque, the robot's tau_J_d (in simulation,
          the last command); tau, the torque sent, after the rate limit and clip;
          robot_mode (-1 in simulation)
        - the controller's attributes as the cycle used them: q_desired, ee_desired, kp,
          kd, ee_kp, ..., mode, and the parameters of registered control laws
        - every number of the robot state (NaN in simulation): tau_J, tau_J_d, dtau_J, q_d,
          dq_d, theta, dtheta, tau_ext_hat_filtered, O_F_ext_hat_K, K_F_ext_hat_K,
          control_command_success_rate, O_T_EE (flat and column-major, as in pylibfranka), ...

        Args:
            fields (list): Fields to record (default: aiofranka.native.RECORD_FIELDS)
            seconds (float): Size of the buffer in C++, in seconds of cycles
            path (str or Path): An .npz file to save the rows to when the recording stops

        Returns:
            Recording: The rows; stop() it, or use it in a with block

        Raises:
            ValueError: If a field is unknown
            RuntimeError: If another recording is running

        Example:
            >>> with controller.record(["time", "q", "tau", "tau_J_d"], path="control.npz"):
            ...     await run_policy(controller)
        """
        current = self._recording
        if current is not None and current.recording:
            raise RuntimeError("A recording is running already; stop() it first")
        if isinstance(fields, str):
            raise TypeError("fields is a list of field names")
        names = list(RECORD_FIELDS if fields is None else fields)
        table = self._record_fields()
        unknown = [name for name in names if name not in table]
        if unknown:
            raise ValueError(f"Cannot record {', '.join(map(repr, unknown))}. Fields: {', '.join(sorted(table))}")
        if len(set(names)) != len(names):
            raise ValueError("A field is listed twice")
        slices, formats, offsets, row = [], [], [], 0
        for name in names:
            source, offset, shape, kind = table[name]
            size = 8 * int(np.prod(shape, dtype=int))
            slices.append((source, offset, size))
            formats.append((np.dtype(kind), shape) if shape else np.dtype(kind))
            offsets.append(row)
            row += size
        dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": row})
        recorder = self._native.Recorder(slices, max(round(seconds * 1000), 1))
        recording = Recording(self._loop, recorder, names, dtype, path)
        self._loop.set_recorder(recorder)
        self.__dict__["_recording"] = recording
        return recording

    def _record_fields(self):
        """{field: (source, offset, shape, dtype)} of what record() can take."""
        native = self._native
        sources = native.RECORD_SOURCES
        table = {}
        for name, offset, shape, kind in native.ROBOT_LAYOUT:
            table[name] = (sources["robot"], offset, tuple(shape), kind)
        for name, offset, shape, kind in native.PARAMS_LAYOUT:
            if name != "custom":
                table[name] = (sources["params"], offset, tuple(shape), kind)
        for name, (offset, shape) in self._custom.items():
            table[name] = (sources["params"], offset, tuple(shape), "f8")
        for name, offset, shape, kind in native.STATE_LAYOUT:
            table[name] = (sources["state"], offset, tuple(shape), kind)
        table["q"], table["dq"] = table["qpos"], table["qvel"]  # the robot's, in its names
        for name, offset, shape, kind in native.RECORD_EXTRA_LAYOUT:
            table[name] = (sources["extra"], offset, tuple(shape), kind)
        return table

    def _drain_recording(self):
        recording = self._recording
        if recording is not None:
            recording._drain()

    # ── The loop ──────────────────────────────────────────────────────────────

    def _configure(self):
        """Give the loop its own copies of the model, and of the data it computes with."""
        robot = self.robot
        model = robot.model
        if not (model.nq == model.nv == model.nu == 7):
            raise ValueError("The native loop needs a model with 7 joints and 7 actuators")
        model = copy.copy(model)
        # In simulation, the loop's data is the simulated arm, continuing from robot.data.
        data = mujoco.MjData(model) if robot.real else copy.copy(robot.data)
        site = robot.site_id
        arrays = {
            "data": data._address,
            "qpos": data.qpos.ctypes.data,
            "qvel": data.qvel.ctypes.data,
            "ctrl": data.ctrl.ctypes.data,
            "site_xpos": data.site_xpos[site].ctypes.data,
            "site_xmat": data.site_xmat[site].ctypes.data,
        }
        self._loop.set_mujoco(model._address, arrays, site, float(model.opt.timestep),
                              float(robot.data.time), _mujoco_functions())
        self.__dict__.update(_loop_model=model, _loop_data=data, _models={},
                             _configured=True, _synced_cycle=-1)
        # RobotInterface asks the loop for states while it runs, and tells it about payloads.
        robot._native_ref = weakref.ref(self)

    def _launch(self, macos_qos=None, cpu=None, fifo_priority=None):
        linux = sys.platform.startswith("linux")
        if macos_qos is None:
            macos_qos = os.environ.get("LIBFRANKA_MACOS_BUSY_WAIT") != "0"
        if fifo_priority is None:
            fifo_priority = self.realtime_priority if linux else 0
        if cpu is None:
            cpu = self.realtime_cpu if linux else None
        cpu = -1 if cpu is None else int(cpu)
        self._configure()
        robot = self.robot
        self._loop.start(robot.torque_controller if robot.real else None,
                         macos_qos=macos_qos, cpu=cpu, fifo_priority=fifo_priority)

    def _owns_connection(self):
        """Whether the loop runs, and so reads every robot state."""
        return self._loop.running()

    def _model_changed(self):
        """RobotInterface.sync_payload() changed the model: give the loop a copy of it."""
        if not self._configured:
            return
        model = copy.copy(self.robot.model)
        epoch = self._loop.swap_model(model._address)
        self._models[epoch] = model

    def _sync_world(self):
        """Bring robot.data and robot.robot_state to the loop's last cycle."""
        if not self._configured:
            return False
        with self._world_lock:
            robot = self.robot
            data = robot.data
            info = self._loop.sync_world(data.qpos, data.qvel, data.ctrl)
            if info is None:
                return False
            cycle, world_time, epoch = info
            if cycle == self._synced_cycle:
                return True
            self.__dict__["_synced_cycle"] = cycle
            if not robot.real:
                data.time = world_time
            mujoco.mj_forward(robot.model, data)
            if robot.real:
                state = self._loop.robot_state()
                if state is not None:
                    robot.robot_state = state
                elif robot.robot_state is not None:
                    self._loop.copy_robot_state_into(robot.robot_state)
            # Model copies the loop has moved past
            for old in [e for e in self._models if e < epoch]:
                del self._models[old]
            return True

    def _tick(self):
        """Called after each sync while the loop runs (the server writes shared memory)."""

    async def _watch(self):
        """Mirror the loop into robot.data until it ends; returns its error, or None."""
        loop = self._loop
        robot = self.robot
        viewer = None if robot.real else getattr(robot, "viewer", None)
        last_stats = last_viewer = last_drain = time.perf_counter()
        status_checked = False
        try:
            while True:
                alive = loop.running()
                self._sync_world()
                self._tick()
                now = time.perf_counter()
                if now - last_drain >= self._DRAIN_PERIOD:
                    self._drain_recording()
                    last_drain = now
                if not status_checked and now - last_stats > 0.05:
                    status_checked = True
                    if loop.realtime_status():
                        logger.warning(f"Native control loop: {loop.realtime_status().rstrip('; ')}")
                if self.track and now - last_stats >= 1.0:
                    self._print_stats()
                    last_stats = now
                if viewer is not None and now - last_viewer >= self._VIEWER_PERIOD:
                    viewer.sync()
                    last_viewer = now
                if not alive:
                    break
                await asyncio.sleep(0.001)
        finally:
            loop.request_stop()
            loop.join()
            self._sync_world()
            self._drain_recording()
        return loop.error() or None

    def loop_stats(self, reset=False):
        """
        Timing of the native loop since start, or since the last call with reset.

        Args:
            reset (bool): Start a new window

        Returns:
            dict: count, mean, std, min and max of the period between cycles [s]; busy_mean
                and busy_max, from a cycle's start to its torque command [s]; what the robot
                reported (real robot only): robot_gap_max [s], the states missed in between,
                and success_min, its lowest command success rate. Since start: warn and error,
                periods off 1 ms by more than 0.1 ms or longer than 10 ms, and max_all [s].
        """
        return self._loop.stats(reset=reset)

    def _print_stats(self):
        stats = self._loop.stats(reset=True)
        if stats["count"] == 0:
            return
        mean, std = stats["mean"] * 1000, stats["std"] * 1000
        low, high = stats["min"] * 1000, stats["max"] * 1000
        print(f"Control loop stats (last {stats['count']} iterations):")
        print(f"  Frequency: {1.0 / stats['mean']:.1f} Hz (target: 1000 Hz)")
        print(f"  Mean dt: {mean:.3f} ms, Std: {std:.3f} ms")
        print(f"  Min dt: {low:.3f} ms, Max dt: {high:.3f} ms")
        print(f"  Jitter (max-min): {high - low:.3f} ms")

    async def _run(self):
        """Watch the native loop; on an error, report it and exit, as FrankaController does."""
        try:
            error = await self._watch()
            if error:
                raise RuntimeError(error)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._report(str(e))
            sys.exit(1)  # Kill the entire script

    def _report(self, error_str):
        """Print the loop's error, save a recording that has a path, and call error_callback."""
        print(f"Error in control loop: {error_str}")
        recording = self._recording
        if recording is not None and recording.recording and recording.path is not None:
            try:
                recording.stop()
                print(f"Saved the recording to {recording.path}")
            except Exception as save_err:
                print(f"Could not save the recording to {recording.path}: {save_err}")
        if self.error_callback is not None:
            try:
                self.error_callback(error_str)
            except Exception as cb_err:
                print(f"Error in error_callback: {cb_err}")

    async def start(self):
        """
        Start the native 1 kHz control loop (FrankaController.start()).

        Returns:
            asyncio.Task: Watches the loop, and ends with it
        """
        logger.info("Starting robot control loop (native)")
        self.robot.start()
        if self.task is None or self.task.done():
            self._launch()
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(1)  # Yield to ensure the task starts
        return self.task

    def step(self):
        """
        Run one control cycle in the calling thread (FrankaController.step()).

        For stepping a simulation, or the robot, by hand while the loop is not running.
        """
        if self._loop.running():
            raise RuntimeError("The native loop is running; it steps by itself")
        if not self._configured:
            self._configure()
        robot = self.robot
        self._loop.step_once(robot.torque_controller if robot.real else None)
        self._sync_world()
        self._drain_recording()
        viewer = None if robot.real else getattr(robot, "viewer", None)
        if viewer is not None:
            viewer.sync()

    # ── FrankaController methods that read robot.data ─────────────────────────

    def initialize(self):
        self._sync_world()
        super().initialize()

    def set_tcp(self, transform):
        self._sync_world()
        super().set_tcp(transform)

    def activate(self, config, check_tool=True, check_null_target=True):
        self._sync_world()
        return super().activate(config, check_tool=check_tool, check_null_target=check_null_target)

    async def move(self, qpos=[0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853]):
        self._sync_world()
        await super().move(qpos)
