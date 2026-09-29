Quick Start
===========

This guide will get you controlling a Franka robot in minutes. There are two ways to use aiofranka:

- **Server mode** — sync Python API, 1kHz loop in a subprocess
- **Async mode** — everything in a single process using ``asyncio``

Server Mode
-------------------------

Server mode runs the 1kHz control loop in a subprocess. Your scripts use a plain synchronous API — no ``async``/``await`` needed.

- **No async/await** — plain Python scripts, easy to integrate with existing codebases
- **Process-isolated** — heavy computation (policy inference, camera processing) can't starve the 1kHz loop
- **Automatic lifecycle** — server subprocess starts with your script and stops when it exits

.. code-block:: python

   import numpy as np
   import aiofranka
   from aiofranka import FrankaRemoteController

   # 1. Unlock the robot (opens brakes + activates FCI)
   aiofranka.unlock()

   # 2. Create controller and start server subprocess
   controller = FrankaRemoteController()
   controller.start()

   # 3. Move to home position
   controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

   # 4. Switch to impedance control
   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0
   controller.kd = np.ones(7) * 4.0
   controller.set_freq(50)

   # 5. Execute sinusoidal motion
   for cnt in range(100):
       state = controller.state
       delta = np.sin(cnt / 50.0 * np.pi) * 0.1
       controller.set("q_desired", delta + controller.initial_qpos)

   # 6. Stop server and lock robot
   controller.stop()
   aiofranka.lock()

The server subprocess terminates automatically when your script exits (Ctrl+C, crash, etc.).
``controller.start()`` checks that the robot is unlocked and FCI is active before launching — if not, it prints a status summary and exits cleanly.

Async Mode
----------

Async mode runs the 1kHz control loop in-process using asyncio — everything in a single script.

- **Single script** — no separate server process, simpler deployment
- **Direct access** — no IPC overhead, full control over the event loop
- **Requires async discipline** — any blocking call >1ms after ``controller.start()`` will cause ``communication_constraints_violation`` (see :doc:`async_mode`)

.. code-block:: python

   import asyncio
   import numpy as np
   from aiofranka import RobotInterface, FrankaController

   async def main():
       robot = RobotInterface("172.16.0.2")
       controller = FrankaController(robot)

       await controller.start()
       await controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

       controller.switch("impedance")
       controller.kp = np.ones(7) * 80.0
       controller.kd = np.ones(7) * 4.0
       controller.set_freq(50)

       for cnt in range(100):
           delta = np.sin(cnt / 50.0 * np.pi) * 0.1
           init = controller.initial_qpos
           await controller.set("q_desired", delta + init)

       await controller.stop()

   if __name__ == "__main__":
       asyncio.run(main())

Server Mode vs Async Mode
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * -
     - Server mode
     - Async mode
   * - **Class**
     - ``FrankaRemoteController``
     - ``FrankaController``
   * - **API style**
     - Synchronous (plain Python)
     - ``async``/``await``
   * - **1kHz loop runs in**
     - Subprocess (auto-managed)
     - Your process (asyncio task)
   * - **Blocking calls OK?**
     - Yes — can't starve the loop
     - No — must stay under ~1ms
   * - **State reads**
     - Shared memory (zero-copy)
     - Direct attribute access
   * - **Commands**
     - ZMQ IPC (msgpack)
     - Direct method calls
   * - **Setup**
     - ``unlock()`` + ``ctrl.start()``
     - Single script
   * - **Best for**
     - Heavy workloads (GPU inference, vision pipelines)
     - Lightweight scripts, rapid prototyping

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

Robot state is continuously updated at 1kHz and accessible via ``controller.state``:

.. code-block:: python

   state = controller.state

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

Test your code without hardware (async mode only):

.. code-block:: python

   import asyncio
   from aiofranka import RobotInterface, FrankaController

   async def test_in_simulation():
       robot = RobotInterface(None)  # None = simulation mode
       controller = FrankaController(robot)

       await controller.start()
       await controller.move()
       await controller.stop()

   asyncio.run(test_in_simulation())

The MuJoCo viewer will open automatically, showing the robot motion.

Next Steps
----------

- :doc:`controllers` — detailed documentation for all control modes
- :doc:`cli` — CLI reference for setup and diagnostics
- :doc:`async_mode` — async discipline guide (for async mode users)
- :doc:`examples` — complete working examples
