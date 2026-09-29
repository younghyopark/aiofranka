Examples
========

This page provides complete, working examples for common use cases.
Examples are shown in both server mode (sync) and async mode where applicable.

Example 1: Simple Motion (Server Mode)
---------------------------------------

Move the robot through positions using the sync API:

.. code-block:: python

   import numpy as np
   import aiofranka
   from aiofranka import FrankaRemoteController

   aiofranka.unlock()

   controller = FrankaRemoteController()
   controller.start()

   # Define waypoints
   home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
   pose1 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

   # Move through waypoints
   for pose in [home, pose1, home]:
       print(f"Moving to: {pose}")
       controller.move(pose)

   controller.stop()
   aiofranka.lock()

Example 2: Simple Motion (Async Mode)
--------------------------------------

Same motion using the async API:

.. code-block:: python

   import asyncio
   import numpy as np
   from aiofranka import RobotInterface, FrankaController

   async def simple_motion():
       robot = RobotInterface("172.16.0.2")
       controller = FrankaController(robot)

       await controller.start()

       try:
           home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
           pose1 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

           for pose in [home, pose1, home]:
               print(f"Moving to: {pose}")
               await controller.move(pose)
               await asyncio.sleep(1.0)

       finally:
           await controller.stop()

   if __name__ == "__main__":
       asyncio.run(simple_motion())

Example 3: Impedance Control
------------------------------

Compliant joint-space control with sinusoidal motion:

.. code-block:: python

   import numpy as np
   import aiofranka
   from aiofranka import FrankaRemoteController

   aiofranka.unlock()

   controller = FrankaRemoteController()
   controller.start()

   # Move to start position
   controller.move()

   # Configure impedance control
   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0
   controller.kd = np.ones(7) * 4.0
   controller.set_freq(50)

   # Execute smooth sinusoidal motion
   for i in range(200):  # 4 seconds at 50 Hz
       delta = np.sin(i / 50.0 * np.pi) * 0.1
       target = controller.initial_qpos + delta
       controller.set("q_desired", target)

   controller.stop()
   aiofranka.lock()

Example 4: Operational Space Control
--------------------------------------

Control end-effector position in Cartesian space:

.. code-block:: python

   import numpy as np
   import aiofranka
   from aiofranka import FrankaRemoteController

   aiofranka.unlock()

   controller = FrankaRemoteController()
   controller.start()

   controller.move()

   # Configure OSC
   controller.switch("osc")
   controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
   controller.ee_kd = np.ones(6) * 10.0
   controller.set_freq(50)

   # Circular motion in XY plane
   for i in range(200):
       angle = i / 50.0 * np.pi
       radius = 0.05

       desired_ee = controller.initial_ee.copy()
       desired_ee[0, 3] += radius * np.cos(angle)
       desired_ee[1, 3] += radius * np.sin(angle)

       controller.set("ee_desired", desired_ee)

   controller.stop()
   aiofranka.lock()

Example 5: Data Collection
----------------------------

Collect synchronized robot data during operation:

.. code-block:: python

   import numpy as np
   import time
   import aiofranka
   from aiofranka import FrankaRemoteController

   aiofranka.unlock()

   controller = FrankaRemoteController()
   controller.start()
   controller.move()

   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0
   controller.kd = np.ones(7) * 4.0
   controller.set_freq(50)

   logs = {'qpos': [], 'qvel': [], 'qdes': [], 'ctrl': [], 'ee': []}

   for i in range(200):
       state = controller.state
       logs['qpos'].append(state['qpos'].copy())
       logs['qvel'].append(state['qvel'].copy())
       logs['ctrl'].append(state['last_torque'].copy())
       logs['ee'].append(state['ee'].copy())
       logs['qdes'].append(controller.q_desired.copy())

       delta = np.sin(i / 50.0 * np.pi) * 0.1
       controller.set("q_desired", delta + controller.initial_qpos)

   for key in logs:
       logs[key] = np.array(logs[key])

   np.savez("robot_data.npz", **logs)
   print(f"Saved {len(logs['qpos'])} samples")

   controller.stop()
   aiofranka.lock()

Example 6: Gain Tuning
------------------------

Systematically test different controller gains:

.. code-block:: python

   import numpy as np
   import time
   import os
   import aiofranka
   from aiofranka import FrankaRemoteController

   aiofranka.unlock()

   controller = FrankaRemoteController()
   controller.start()

   base = np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])
   kps = [16, 32, 64, 128, 256]
   kds = [1, 2, 4, 8, 16]

   for kp in kps:
       for kd in kds:
           # Move to start
           controller.kp = base * 80
           controller.kd = base * 4
           controller.move()
           time.sleep(1.0)

           print(f"Testing kp={kp}, kd={kd}")
           controller.switch("impedance")
           controller.kp = base * kp
           controller.kd = base * kd
           controller.set_freq(50)

           logs = {'qpos': [], 'qdes': []}
           for cnt in range(200):
               state = controller.state
               logs['qpos'].append(state['qpos'].copy())
               logs['qdes'].append(controller.q_desired.copy())

               delta = np.sin(cnt / 50.0 * np.pi) * 0.1
               controller.set("q_desired", delta + controller.initial_qpos)

           for key in logs:
               logs[key] = np.stack(logs[key])

           os.makedirs("sysid_data", exist_ok=True)
           np.savez(f"sysid_data/K{kp}_D{kd}.npz", **logs)

   controller.stop()
   aiofranka.lock()

Example 7: End-Effector Configuration
---------------------------------------

Set end-effector mass and center of mass for accurate gravity compensation:

.. code-block:: python

   import aiofranka

   aiofranka.unlock()

   # Set end-effector parameters (mass in kg, CoM in meters)
   aiofranka.set_configuration(mass=1.0, com=[0, 0, 0.057])

   # Now start your control script...
   # The robot will use the updated parameters for gravity compensation

   aiofranka.lock()

Example 8: Gripper Control
---------------------------

Control a Robotiq gripper alongside the robot arm:

.. code-block:: python

   import asyncio
   from aiofranka import GripperController

   async def gripper_demo():
       gripper = GripperController("/dev/ttyUSB1")
       await gripper.start()

       # Set speed and force (like kp/kd for the arm)
       gripper.speed = 128
       gripper.force = 200

       # Open and close
       gripper.q_desired = 0     # Open
       await gripper.wait_until_reached()

       gripper.q_desired = 255   # Close
       await gripper.wait_until_reached()

       # Or use convenience methods
       gripper.open()
       await gripper.wait_until_reached()

       gripper.close()
       await gripper.wait_until_reached()

       await gripper.stop()

   asyncio.run(gripper_demo())

.. note::
   Gripper support requires: ``pip install "aiofranka[robotiq]"``

Example 9: Simulation Testing
-------------------------------

Test your controller in simulation before deploying to real robot (async mode only):

.. code-block:: python

   import asyncio
   import numpy as np
   from aiofranka import RobotInterface, FrankaController

   async def test_algorithm(robot_ip=None):
       robot = RobotInterface(robot_ip)
       controller = FrankaController(robot)

       await controller.start()

       try:
           mode = "SIMULATION" if robot_ip is None else "REAL"
           print(f"Testing in {mode} mode")

           await controller.move()

           controller.switch("impedance")
           controller.kp = np.ones(7) * 80.0
           controller.kd = np.ones(7) * 4.0
           controller.set_freq(50)

           for i in range(100):
               delta = np.sin(i / 50.0 * np.pi) * 0.1
               target = controller.initial_qpos + delta
               await controller.set("q_desired", target)

           print("Test successful!")

       finally:
           await controller.stop()

   if __name__ == "__main__":
       # First test in simulation
       asyncio.run(test_algorithm(None))

       # Then deploy to real robot
       asyncio.run(test_algorithm("172.16.0.2"))

More Examples
-------------

The ``examples/`` directory in the repository has minimal async mode scripts. Each takes the robot
IP as an optional argument and runs in MuJoCo without it:

- ``00_move.py``: Move one joint (``--joint``, ``--delta``)
- ``01_joint_impedance.py``: Hold the current joint positions with joint impedance control
- ``02_osc_hold.py``: Hold the current end-effector pose with operational space control
- ``03_zero_torque.py``: Stream zero torque, so the robot only compensates gravity

Research scripts (system identification, SpaceMouse teleoperation, Robotiq gripper) are on the
`research-scripts <https://github.com/younghyopark/aiofranka/tree/research-scripts>`_ branch.

Next Steps
----------

- Review :doc:`controllers` for detailed controller documentation
- Check :doc:`cli` for robot setup commands
- Explore the :doc:`async_mode` guide if using async mode
