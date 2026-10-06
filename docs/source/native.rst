Native Control Loop
===================

aiofranka's 1 kHz control loop runs in C++, in a thread that never waits for Python.
``Controller`` runs it in your process with plain calls, and ``NativeFrankaController`` with
awaitable ones, for asyncio code.

The legacy ``FrankaController`` runs the loop on the asyncio event loop, in Python. Anything
else that runs there delays the next torque command: a planner, loading a model, garbage
collection, a thread holding the GIL. When commands are late, the robot stops with
``communication_constraints_violation``. The native controllers have ``FrankaController``'s
methods and attributes, so porting legacy code means swapping the class:

.. code-block:: python

   import aiofranka

   controller = aiofranka.Controller(aiofranka.Robot("172.16.0.2"))
   controller.start()
   controller.switch("osc")
   controller.set("ee_desired", target)

or, keeping the ``await``\ s, ``NativeFrankaController(RobotInterface(ip))`` for
``FrankaController(RobotInterface(ip))``.

Server mode runs the native loop too: ``FrankaRemoteController``, ``aiofranka start-server`` and
``aiofranka.start()`` start the server with ``NativeServerController``, and ``aiofranka home``,
``aiofranka gravcomp`` and ``aiofranka tool identify`` run ``NativeFrankaController``. Where the
native loop is not built, the server, ``home`` and ``gravcomp`` fall back to the Python loop
with a warning. ``aiofranka start-server --python``, ``aiofranka.start(native=False)`` and
``FrankaRemoteController(native=False)`` run the legacy Python loop.

.. contents:: Table of Contents
   :local:
   :depth: 2


How it compares
---------------

The impedance, pid, osc and torque laws are ports of ``FrankaController``'s. Stepped in
lockstep from the same state, impedance, pid and torque mode send bit-identical torques,
and the OSC agrees to 1e-12 Nm.

On a simulated robot driven through libfranka's ``readOnce()``/``writeOnce()``, the longest
gap between two torque commands was:

.. list-table::
   :header-rows: 1

   * - While Python...
     - ``FrankaController``
     - ``NativeFrankaController``
   * - idles
     - 1.3 ms
     - 1.2 ms
   * - blocks the event loop for 300 ms
     - 302 ms
     - 1.3 ms
   * - runs a thread that holds the GIL for 300 ms
     - 18.8 ms
     - 1.2 ms

What changes:

- A subclass's ``step()`` would never run, so the native controllers refuse one: write it
  as a control law, and log every cycle with ``record()`` (both below).
- Attributes the loop reads (``kp``, ``q_desired``, ``ee_desired``, ...) are views of its
  memory. Assigning one copies the value in, and the loop takes it whole at its next cycle.
- While the loop runs, ``robot.state`` is what it read at its last cycle. With
  ``NativeFrankaController``, the ``RobotInterface``'s ``robot.data`` and ``robot.robot_state``
  follow it, updated from the event loop about every millisecond. In simulation, the loop
  owns the simulated arm.
- On Linux, the loop's thread runs at SCHED_FIFO priority 80, which needs an rtprio limit
  (``ulimit -r``) of at least that. Set ``controller.realtime_priority = 0`` before
  ``start()`` for normal priority.
- The thread starts on the CPUs of the thread that calls ``start()``. Set
  ``controller.realtime_cpu`` before ``start()`` to pin it to a CPU of its own, and keep
  Python's threads at normal priority: another SCHED_FIFO thread of the same priority on
  that CPU would hold the loop off until it yields.


Building
--------

The loop is the compiled extension ``aiofranka._native``. The wheels for macOS on Apple
Silicon (CPython 3.10 to 3.14) and Linux x86_64 (CPython 3.10 to 3.12) include it; elsewhere,
or from a git checkout, it is built with a C++17 compiler. For a development install:

.. code-block:: bash

   pip install "pybind11>=3.1,<3.2"
   pip install --no-build-isolation -e .

It is not linked against libfranka: it uses the libfranka that pylibfranka loaded, and must
be rebuilt for another libfranka minor version.


Recording every cycle
---------------------

``controller.record()`` logs every cycle of the loop. After sending the torques, the loop
writes the chosen fields into a buffer in C++ without waiting for Python, and the
controller moves them to Python about every 10 ms, so a blocked event loop loses nothing:

.. code-block:: python

   with controller.record(["time", "q", "dq", "tau", "tau_J_d"], path="control.npz") as recording:
       run_policy(controller)
   data = recording.data()  # {"time": (n,), "q": (n, 7), ...}, one row per cycle

The fields are those of the cycle (``cycle``, ``time``, ``wall_time``, ``busy``, ``q``,
``dq``, ``ee``, ``tcp``, ``jac``, ``mm``, ``last_torque``, ``tau``, ``robot_mode``), the
controller's attributes as the cycle used them (``q_desired``, ``ee_desired``, ``kp``, ...,
and the parameters of registered control laws), and every number of the robot state
(``tau_J``, ``tau_J_d``, ``tau_ext_hat_filtered``, ``O_F_ext_hat_K``,
``control_command_success_rate``, ...; NaN in simulation). ``record()`` without fields
takes ``aiofranka.native.RECORD_FIELDS``.

- ``tau`` is the torque sent, after the rate limit and clip. The robot echoes each command
  it got in the ``tau_J_d`` of its next state, rounded to float32, so the rows show which
  commands arrived.
- ``wall_time`` is the host's ``time.time()`` when the robot state arrived, and ``busy`` the
  seconds from then to the command.
- With a ``path``, ``stop()`` saves the rows there, as does a loop that stops with an error
  before the process exits. ``recording.save(path)`` saves the rows so far.
- The buffer holds ``seconds`` of cycles (default 60). If Python does not take the rows for
  longer, the loop drops new ones and ``recording.dropped`` counts them.


Custom control laws
-------------------

Write a new law in Python. aiofranka compiles it with Numba
(``pip install "aiofranka[native]"``), and the loop calls it every millisecond without
Python:

.. code-block:: python

   from aiofranka import control_law

   @control_law(params={"stiffness": 7, "damping": 7}, memory={"integral": 7})
   def pi_damping(s, p, m, tau):
       error = p.q_desired - s.qpos
       m.integral[:] += error * s.dt
       tau[:] = p.stiffness * error + 2.0 * m.integral - p.damping * s.qvel

   controller.switch(pi_damping)            # compiles it, a few seconds the first time
   controller.stiffness = np.full(7, 60.0)  # its params are controller attributes, zero at first
   controller.damping = np.full(7, 4.0)
   controller.set("q_desired", target)

A law gets:

- ``s``, what the loop read at this cycle: ``qpos``, ``qvel``, ``ee`` (the flange pose,
  4x4), ``jac`` (6x7, linear rows first), ``mm`` (7x7), ``last_torque``, and ``cycle``,
  ``time`` and ``dt``
- ``p``, the controller's attributes (``kp``, ``q_desired``, ``ee_desired``, ...) and the
  law's own ``params``
- ``m``, the law's ``memory``, zeroed when ``switch()`` selects the law
- ``tau``, the 7 torques to fill [Nm]

Laws use numpy and math only (Numba's nopython mode). With ``clip`` (the default), the loop
rate-limits and clips their torques like the built-in laws. Return a nonzero integer to stop
the loop with an error. Try a law in simulation first: ``controller.step()`` runs one cycle
at a time. ``examples/08_native_custom_law.py`` holds the arm with a PID law.
