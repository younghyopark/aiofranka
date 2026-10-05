Native Control Loop
===================

``FrankaController`` runs its 1 kHz control loop on the asyncio event loop, in Python.
Anything else that runs there delays the next torque command: a planner, loading a model,
garbage collection, a thread holding the GIL. When commands are late, the robot stops with
``communication_constraints_violation``.

``NativeFrankaController`` runs the same loop in C++, in a thread that never waits for
Python. It has ``FrankaController``'s constructor, methods and attributes, so porting means
swapping the class:

.. code-block:: python

   from aiofranka import NativeFrankaController, RobotInterface

   robot = RobotInterface("172.16.0.2")
   controller = NativeFrankaController(robot)  # instead of FrankaController(robot)
   await controller.start()
   controller.switch("osc")
   await controller.set("ee_desired", target)

For server mode, swap ``FrankaRemoteController`` for ``FrankaRemoteControllerNative``, or
start the server with ``aiofranka start-server --native`` (``aiofranka.start(native=True)``
from Python).

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

- A subclass's ``step()`` would never run, so ``NativeFrankaController`` refuses one: write
  it as a control law (below).
- Attributes the loop reads (``kp``, ``q_desired``, ``ee_desired``, ...) are views of its
  memory. Assigning one copies the value in, and the loop takes it whole at its next cycle.
- While the loop runs, ``robot.data`` and ``robot.robot_state`` follow it, updated from the
  event loop about every millisecond. In simulation, the loop owns the simulated arm.
- On Linux, the loop's thread runs at SCHED_FIFO priority 80, which needs an rtprio limit
  (``ulimit -r``) of at least that. Set ``controller.realtime_priority = 0`` before
  ``start()`` for normal priority.


Building
--------

The loop is the compiled extension ``aiofranka._native``, built when aiofranka is installed
from source with a C++17 compiler. For a development install:

.. code-block:: bash

   pip install "pybind11>=3.1,<3.2"
   pip install --no-build-isolation -e .

It is not linked against libfranka: it uses the libfranka that pylibfranka loaded, and must
be rebuilt for another libfranka minor version.


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
   await controller.set("q_desired", target)

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
