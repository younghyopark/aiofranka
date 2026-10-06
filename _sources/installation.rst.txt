Installation
============

System Requirements
-------------------

- Linux operating system (Ubuntu 20.04+ recommended)
- Real-time kernel (for best 1kHz control loop performance)
- Network connection to Franka robot
- Python 3.8+

Installing aiofranka
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release:

.. code-block:: bash

   pip install aiofranka

From Source
~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/younghyopark/aiofranka.git
   cd aiofranka
   pip install -e .

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For Robotiq gripper support:

.. code-block:: bash

   pip install "aiofranka[robotiq]"

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import aiofranka
   print(aiofranka.__version__)

   # Test simulation mode (no robot required)
   from aiofranka import RobotInterface
   robot = RobotInterface(None)  # None = simulation
   state = robot.state
   print(f"Joint positions: {state['qpos']}")

If this runs without errors, you're ready to go!

First-Time Robot Setup
----------------------

1. Make sure you can access the Franka Desk GUI from your browser by navigating to the robot's IP (e.g. ``https://172.16.0.2``).

2. Unlock the robot and activate FCI:

   .. code-block:: bash

      aiofranka unlock --ip 172.16.0.2

   You'll be prompted for Franka Desk credentials on first use. They are saved to ``~/.aiofranka/config.json`` for subsequent use.

3. Verify the robot is ready:

   .. code-block:: bash

      aiofranka status
