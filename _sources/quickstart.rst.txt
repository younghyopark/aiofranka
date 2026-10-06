Quick Start
===========

This guide will get you controlling a Franka robot in minutes. Unlock the robot first
(see `Unlocking and Locking`_ below), then:

.. code-block:: python

   import numpy as np
   import aiofranka

   robot = aiofranka.Robot("172.16.0.2")       # None: a simulated arm in MuJoCo
   controller = aiofranka.Controller(robot)

   controller.start()                           # takes torque control, starts the 1 kHz loop
   controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0
   controller.kd = np.ones(7) * 4.0
   controller.set_freq(50)

   for cnt in range(100):
       delta = np.sin(cnt / 50.0 * np.pi) * 0.1
       controller.set("q_desired", controller.initial_qpos + delta)   # sleeps to hold 50 Hz

   print(robot.state["qpos"])
   controller.stop()                            # gives torque control back

``Robot`` is the arm: its connection, its MuJoCo model with the tool, and its state, whoever
drives it. ``Controller`` drives it. Its 1 kHz loop runs in C++, in a thread that never waits
for Python (see :doc:`native`), and its methods are plain calls: your script can block, sleep
or run a policy without delaying a torque command. One controller drives a robot at a time,
and ``with aiofranka.Controller(robot) as controller:`` starts and stops it.

Robot and Controller
--------------------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - ``Robot``: the arm
     - ``Controller``: how it is driven
   * - **Lifecycle**
     - ``Robot(ip)`` connects, ``close()`` disconnects; ``Robot(None)`` is MuJoCo
     - ``start()`` takes torque control and starts the 1 kHz loop; ``stop()`` gives it back
   * - **State**
     - ``robot.state``, whether or not a controller runs; ``robot.robot_state``, libfranka's
     - Gains, targets, mode (``controller.type``), ``loop_stats()``
   * - **Model**
     - The MuJoCo model with the tool (``robot.tool``), ``set_load()``
     - The point the OSC controls, ``set_tcp()``
   * - **Behavior**
     - —
     - ``switch()``, control laws, ``activate(config)``, ``move()``, ``set()``/``set_freq()``,
       ``record()``, ``identify_payload()``

asyncio Code
------------

``NativeFrankaController`` is the same controller with awaitable methods, for programs built
on asyncio:

.. code-block:: python

   controller = aiofranka.NativeFrankaController(aiofranka.RobotInterface("172.16.0.2"))
   await controller.start()
   await controller.move()
   await controller.set("q_desired", target)
   await controller.stop()

A blocked event loop does not delay its torque commands either, but it holds ``set()``,
``move()`` and the copy of the state into ``robot.data`` until it runs again (see
:doc:`async_mode`).

Legacy
------

Earlier versions had two other ways, which keep working but are soft-deprecated:

- **Server mode**, ``FrankaRemoteController``: a sync API with the loop in a subprocess and
  every command an IPC round trip, without ``record()``, control laws, ``set_tcp()``,
  ``activate()`` or ``identify_payload()``. ``Controller`` gives the same plain calls without a
  second process.
- **The Python loop**: ``FrankaController`` on the asyncio event loop, the server with
  ``aiofranka start-server --python`` (or ``aiofranka.start(native=False)``,
  ``FrankaRemoteController(native=False)``), and ``FrankaRemoteControllerV2`` in a Python thread.
  Anything else that runs in Python can delay their torque commands, so any blocking call over
  about 1 ms can stop the robot with ``communication_constraints_violation``.

To port, replace ``FrankaController(RobotInterface(ip))`` or ``FrankaRemoteController(ip)`` with
``Controller(Robot(ip))`` and drop the ``await``\ s; a subclass that overrides ``step()``
becomes a control law (see :doc:`native`).

Unlocking and Locking
---------------------

Before using the robot, joints must be unlocked and FCI (Franka Control Interface) must be activated.

**From Python:**

.. code-block:: python

   import aiofranka

   aiofranka.unlock()   # opens brakes + activates FCI
   # ... run your control script ...
   aiofranka.lock()     # closes brakes + deactivates FCI

**From the CLI:**

.. code-block:: bash

   aiofranka unlock
   # ... run your script ...
   aiofranka lock

Credentials are prompted on first use and saved to ``~/.aiofranka/config.json``.

Reading Robot State
-------------------

The arm's state is ``robot.state``: while a controller drives the arm, what its 1 kHz loop
read at its last cycle; otherwise the arm is read when you ask.

.. code-block:: python

   state = robot.state

   print(f"Joint positions: {state['qpos']}")        # (7,) [rad]
   print(f"Joint velocities: {state['qvel']}")       # (7,) [rad/s]
   print(f"End-effector pose:\n{state['ee']}")       # (4, 4) homogeneous transform
   print(f"Jacobian:\n{state['jac']}")               # (6, 7)
   print(f"Mass matrix:\n{state['mm']}")             # (7, 7)
   print(f"Last torques: {state['last_torque']}")    # (7,) [Nm]

Additional state available on the controller:

.. code-block:: python

   controller.initial_qpos   # (7,) joint positions at last switch()
   controller.initial_ee     # (4, 4) EE pose at last switch()
   controller.q_desired      # (7,) current desired joint positions
   controller.ee_desired     # (4, 4) current desired EE pose

Rate Limiting
-------------

Use ``set_freq()`` to enforce strict timing for command updates:

.. code-block:: python

   controller.set_freq(50)  # Set 50Hz update rate

   # Automatically sleeps to maintain 50Hz timing
   for i in range(100):
       controller.set("q_desired", compute_target())

Each ``set()`` call sleeps for the remainder of the period, so the loop maintains consistent timing even if your computation time varies.

Simulation Mode
---------------

Test your code without hardware:

.. code-block:: python

   import aiofranka

   robot = aiofranka.Robot(None)  # None = simulation mode
   with aiofranka.Controller(robot) as controller:
       controller.move()

The MuJoCo viewer will open automatically, showing the robot motion. On macOS, run the script
with ``mjpython``, which MuJoCo's viewer needs there.

Next Steps
----------

- :doc:`controllers` — detailed documentation for all control modes
- :doc:`cli` — CLI reference for setup and diagnostics
- :doc:`native` — the native control loop, custom control laws and recording
- :doc:`async_mode` — keeping the event loop responsive with ``NativeFrankaController``
- :doc:`examples` — complete working examples
