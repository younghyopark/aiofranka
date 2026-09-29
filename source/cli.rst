CLI Reference
=============

The CLI handles robot setup, server lifecycle, and diagnostics.

.. code-block:: text

   aiofranka start-server [--ip IP] [--no-home]  Start the control server
   aiofranka unlock   [--ip IP]              Unlock joints + activate FCI
   aiofranka lock     [--ip IP]              Lock joints + deactivate FCI
   aiofranka gravcomp [--ip IP] [--damping]  Gravity compensation (freedrive)
   aiofranka home     [--ip IP]              Move the robot to its home pose
   aiofranka status   [--ip IP]              Show robot & server status
   aiofranka stop     [--ip IP]              Stop a running server
   aiofranka mode     [--ip IP] [--set MODE] View/change operating mode
   aiofranka config   [--ip IP] [--mass M]   View/set end-effector config
   aiofranka selftest [--ip IP] [--force]    Run safety self-tests
   aiofranka log      [-n LINES] [-f]        View server logs
   aiofranka gripper  --open|--close          Control the Robotiq gripper
   aiofranka rt-benchmark [--duration SEC]    Benchmark the 1 kHz control loop

unlock / lock
-------------

Unlock opens the brakes and activates FCI so the robot is ready for torque control.
Lock does the reverse. Credentials are prompted on first use and saved to ``~/.aiofranka/config.json``.

.. code-block:: bash

   # Unlock before running your script
   aiofranka unlock

   # Lock when you're done
   aiofranka lock

You can also do this from Python:

.. code-block:: python

   import aiofranka
   aiofranka.unlock()   # opens brakes + activates FCI
   # ... run your control script ...
   aiofranka.lock()     # closes brakes + deactivates FCI

gravcomp
--------

Runs gravity compensation mode in the foreground. The robot is freely movable by hand.
Press Ctrl+C to stop control; the joints remain unlocked with FCI active.
Run ``aiofranka lock`` when finished.

.. code-block:: bash

   aiofranka gravcomp                  # default: zero damping
   aiofranka gravcomp --damping 2.0    # add velocity damping

status
------

Shows robot state (joints locked/unlocked, FCI active/inactive, control token,
self-test status, end-effector configuration) and server status if running.

.. code-block:: bash

   aiofranka status

stop
----

Sends a shutdown signal to a running server process. The server deactivates FCI,
locks joints, and releases the control token.

.. code-block:: bash

   aiofranka stop

mode
----

View or change the operating mode. ``Execution`` is needed for FCI control.
``Programming`` enables freedrive via the pilot interface button near the end-effector.

.. code-block:: bash

   aiofranka mode                  # view current mode
   aiofranka mode --set Execution  # switch to FCI mode

config
------

View or set the end-effector configuration (mass, center of mass, inertia,
flange-to-EE transform). Changes are applied via the Franka Desk API.

.. code-block:: bash

   aiofranka config                                # view current config
   aiofranka config --mass 0.5 --com 0,0,0.03      # set mass + CoM
   aiofranka config --translation 0,0,0.1           # set flange-to-EE offset

You can also set end-effector configuration from Python:

.. code-block:: python

   import aiofranka
   aiofranka.unlock()
   aiofranka.set_configuration(mass=0.5, com=[0, 0, 0.03])
   aiofranka.lock()

selftest
--------

Run the robot's safety self-tests. The robot will lock joints during the test.

.. code-block:: bash

   aiofranka selftest          # run if due
   aiofranka selftest --force  # run even if not due

log
---

View recent server log entries from ``~/.aiofranka/server.log``.

.. code-block:: bash

   aiofranka log              # last 20 lines
   aiofranka log -n 100       # last 100 lines
   aiofranka log -f           # follow (like tail -f)

Common Flags
------------

Most commands accept these flags:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--ip IP``
     - Robot IP address (default: last used, or ``172.16.0.2``)
   * - ``--username USER``
     - Franka Desk web UI username (default: saved or prompted)
   * - ``--password PASS``
     - Franka Desk web UI password (default: saved or prompted)
   * - ``--protocol http|https``
     - Web UI protocol (default: ``https``)
