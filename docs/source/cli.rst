CLI Reference
=============

The CLI handles robot setup, server lifecycle, and diagnostics.

.. code-block:: text

   aiofranka start-server [--ip IP] [--no-home]  Start the control server
   aiofranka unlock   [--ip IP]              Unlock joints + activate FCI
   aiofranka lock     [--ip IP]              Lock joints + deactivate FCI
   aiofranka gravcomp [--ip IP] [--mode program]  Move the robot by hand (freedrive)
   aiofranka home     [--ip IP]              Move the robot to its home pose
   aiofranka status   [--ip IP]              Show robot & server status
   aiofranka stop     [--ip IP]              Stop a running server
   aiofranka mode     [--ip IP] [program|execute]  View/change operating mode
   aiofranka config   [--ip IP] [--mass M]   View/set end-effector config
   aiofranka tool     identify|load|list ... Identify, save and load tools
   aiofranka camera   calibrate|fit          Locate a fixed camera relative to the robot
   aiofranka selftest [--ip IP] [--force]    Run safety self-tests
   aiofranka log      [-n LINES] [-f]        View server logs
   aiofranka gripper  --open|--close          Control the Robotiq gripper
   aiofranka rt-benchmark [--duration SEC]    Benchmark the 1 kHz control loop

unlock / lock
-------------

Unlock opens the brakes and activates FCI so the robot is ready for torque control.
It first recovers safety errors, runs the self-tests if they are overdue, and switches
from Programming back to Execution. Lock does the reverse. Credentials are prompted on first use and saved to ``~/.aiofranka/config.json``.

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

With ``--mode program``, it switches the robot to Programming mode instead, as
``aiofranka mode program`` does: the arm moves by hand only while the guiding button on the end
effector is held, and nothing keeps running. ``aiofranka unlock`` switches back for FCI.

.. code-block:: bash

   aiofranka gravcomp                  # default: zero damping
   aiofranka gravcomp --damping 2.0    # add velocity damping
   aiofranka gravcomp --mode program   # hand-guide with the guiding button instead

status
------

Shows robot state (joints locked/unlocked, FCI active/inactive, control token,
self-test status, end-effector configuration) and server status if running.

.. code-block:: bash

   aiofranka status

start-server
------------

Runs the control server in the background for clients such as ``FrankaRemoteController``
(``--foreground`` keeps it in the terminal). It unlocks the robot (``--no-unlock`` skips that),
moves it home (``--no-home`` skips that) and runs the 1 kHz loop in C++
(``NativeServerController``). ``--python`` runs the legacy Python loop instead.

.. code-block:: bash

   aiofranka start-server              # the native loop
   aiofranka start-server --python     # the legacy Python loop

stop
----

Sends a shutdown signal to a running server process. The server deactivates FCI,
locks joints, and releases the control token.

.. code-block:: bash

   aiofranka stop

mode
----

View or change the operating mode. ``Execution`` is needed for FCI control.
``Programming`` enables freedrive via the pilot interface button near the end-effector,
as Desk's mode switch does: ``aiofranka mode program`` deactivates FCI, opens the brakes
if they are closed, and hands the control token back to Desk. ``aiofranka unlock``
switches back to Execution by itself.

.. code-block:: bash

   aiofranka mode            # view current mode
   aiofranka mode program    # switch to Programming, to hand-guide the robot
   aiofranka mode execute    # switch back to Execution, for FCI

config
------

View or set the end-effector configuration (mass, center of mass, inertia,
flange-to-EE transform). Changes are applied via the Franka Desk API to the active
end-effector profile; to keep several tools by name, use ``aiofranka tool`` below.

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

tool
----

Identify the tool on the flange and keep it as an end-effector profile in Desk (Settings > End
Effector), the same profiles the web UI shows. The robot compensates the active profile, and
``RobotInterface`` merges it into the MuJoCo model when it connects.

.. code-block:: bash

   aiofranka tool identify gripper   # move through 16 poses, save as profile "gripper", activate it
   aiofranka tool load gripper       # activate a profile
   aiofranka tool unload             # activate the built-in "No End Effector" profile
   aiofranka tool list               # list the profiles, marking the active one
   aiofranka tool remove gripper     # delete a profile

``tool identify`` unlocks the robot like ``aiofranka home`` and plans the poses around the current
one, so move the arm to an open pose first. Describe the tool for the collision checks with
``--tool-length`` and ``--tool-radius`` (default 0.2 m and 0.1 m), and the table with ``--floor``
(default 0, the mounting plane; the robot keeps 5 cm from it). It shows the estimate and
asks before saving it, offering to edit the mass (in g) and the center of mass (in mm), e.g. to use
a scale reading; an existing profile keeps its tool center point and inertia. ``--no-load`` saves
it without activating it. See :ref:`payload-identification`.

From Python:

.. code-block:: python

   import aiofranka
   aiofranka.save_tool("gripper", mass=0.62, com=[0.0, 0.0, 0.045])
   aiofranka.load_tool("gripper")
   aiofranka.list_tools()

camera
------

Locate a fixed camera in the robot's base frame with an AprilCube held on the flange. Print
aprilcube's calibration cube
(`cube.3mf <https://github.com/younghyopark/aprilcube/blob/main/models/calibration_cube/cube.3mf>`_),
mount it with its connector, start the camera with ``aiocamera start``, and make the cube's mass the
active Desk profile. It needs the camera extra: ``pip install "aiofranka[camera]"``.

.. code-block:: bash

   aiofranka camera calibrate      # move the arm by hand; captures and fits into camera_calibration/<date>/
   aiofranka camera fit SESSION    # fit a recorded session again

``camera calibrate`` switches the robot to Programming mode like ``aiofranka mode program``, with a
live view of the camera image in the terminal. Move the arm holding the guiding button on the end
effector, and let go: whenever it has rested for 0.7 s at a new pose, 5 cm or 10 deg from every view
so far, with the cube in view, it records the view and beeps. The joint positions come from Desk, as
its web UI gets them. Space captures anyway, ``u`` removes the last view, Enter fits and ``q`` quits
keeping the views. The fit writes ``calibration.json`` with ``T_base_camera``, ``T_ee_cube``, the
intrinsics and the reprojection errors on held-out views. The robot stays in Programming mode;
``aiofranka unlock`` switches back.

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

rt-benchmark
------------

Measures the 1 kHz loop on this machine. It holds the current pose in gravcomp, impedance and
OSC in turn, 10 s each (``--duration``), records every cycle and compares the modes. ``--mode``
picks the modes; one mode prints its full report: the periods and their percentiles, the
response time from ``readOnce`` to ``writeOnce`` against a 300 us budget, skipped robot
states, dropped commands and a histogram. ``--python`` benchmarks the legacy Python loop
instead, with a breakdown per phase, and ``--all-combos`` compares its real-time settings.

.. code-block:: bash

   aiofranka rt-benchmark                 # the native loop, every mode
   aiofranka rt-benchmark --mode osc      # one mode, its full report
   aiofranka rt-benchmark --python        # the legacy Python loop

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
