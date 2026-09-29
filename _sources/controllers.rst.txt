Controllers
===========

aiofranka supports four control modes, each suited for different applications.
All modes work with both server mode (``FrankaRemoteController``) and async mode (``FrankaController``).

.. contents:: Table of Contents
   :local:
   :depth: 2


Impedance Control
-----------------

Joint-space impedance control implements a spring-damper system:

.. math::

   \tau = K_{\text{p}} (\mathbf{q}_{\text{desired}} - \mathbf{q}) - K_{\text{d}} \dot{\mathbf{q}}

where:

- :math:`\tau`: Joint torques [Nm]
- :math:`K_{\text{p}}`: Position stiffness gains [Nm/rad] — ``controller.kp``
- :math:`K_{\text{d}}`: Damping gains [Nm·s/rad] — ``controller.kd``
- :math:`\mathbf{q}_{\text{desired}}`: Desired joint positions [rad] — ``controller.q_desired``
- :math:`\mathbf{q}`: Current joint positions [rad]
- :math:`\dot{\mathbf{q}}`: Joint velocities [rad/s]

Torque rate limiting is applied when ``controller.clip = True`` (default).

Usage (server mode)
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0  # Stiffness
   controller.kd = np.ones(7) * 4.0   # Damping
   controller.set_freq(50)

   for i in range(200):
       target = compute_target(i)
       controller.set("q_desired", target)

Usage (async mode)
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0
   controller.kd = np.ones(7) * 4.0
   controller.set_freq(50)

   for i in range(200):
       target = compute_target(i)
       await controller.set("q_desired", target)

**Best for**: Joint-space trajectories, compliant behavior, system identification.

**Default gains**: ``kp = 80``, ``kd = 4`` (per joint).


PID Control
-----------

Joint-space PID control with integral term for reducing steady-state error:

.. math::

   \tau = K_{\text{p}} \mathbf{e} + K_{\text{i}} \int \mathbf{e} \, dt - K_{\text{d}} \dot{\mathbf{q}}

where :math:`\mathbf{e} = \mathbf{q}_{\text{desired}} - \mathbf{q}`.

The integral term is clamped (anti-windup) to prevent unbounded growth.
Integral state is reset when switching controllers via ``switch()``.

Usage
~~~~~

.. code-block:: python

   controller.switch("pid")
   controller.kp = np.ones(7) * 80.0   # Proportional
   controller.ki = np.ones(7) * 0.1    # Integral
   controller.kd = np.ones(7) * 4.0    # Derivative (damping)
   controller.set_freq(50)

   for i in range(200):
       target = compute_target(i)
       controller.set("q_desired", target)  # or await in async mode

**Best for**: Tasks requiring zero steady-state error, precise positioning.

**Default gains**: ``kp = 80``, ``ki = 0.1``, ``kd = 4`` (per joint).


Operational Space Control (OSC)
-------------------------------

OSC controls the end-effector in Cartesian space while managing null-space behavior:

.. math::

   \tau = \mathbf{J}^T \mathbf{M}_{\mathbf{x}} (K_{\text{p}}^{\text{ee}} \mathbf{e} - K_{\text{d}}^{\text{ee}} \dot{\mathbf{x}}) + (\mathbf{I} - \mathbf{J}^T \bar{\mathbf{J}}^T) (K_{\text{p}}^{\text{null}} (\mathbf{q}_0 - \mathbf{q}) - K_{\text{d}}^{\text{null}} \dot{\mathbf{q}})

where:

- :math:`\mathbf{J}`: End-effector Jacobian (6x7) — ``state['jac']``
- :math:`\mathbf{M}_{\mathbf{x}} = (\mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T)^{-1}`: Operational space inertia matrix
- :math:`\mathbf{M}`: Joint-space mass matrix (7x7) — ``state['mm']``
- :math:`\mathbf{e} = [\mathbf{p}_{\text{goal}} - \mathbf{p}; \text{Log}(\mathbf{R}_{\text{goal}} \mathbf{R}^{-1})]`: Pose error (position + rotation as axis-angle)
- :math:`\dot{\mathbf{x}} = \mathbf{J} \dot{\mathbf{q}}`: End-effector velocity
- :math:`\bar{\mathbf{J}} = \mathbf{M}^{-1} \mathbf{J}^T \mathbf{M}_{\mathbf{x}}`: Dynamically consistent pseudoinverse
- :math:`(\mathbf{I} - \mathbf{J}^T \bar{\mathbf{J}}^T)`: Null-space projection matrix
- :math:`\mathbf{q}_0`: Null-space reference configuration (``controller.initial_qpos``)

Usage
~~~~~

.. code-block:: python

   controller.switch("osc")

   # Task-space gains [x, y, z, roll, pitch, yaw]
   controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
   controller.ee_kd = np.ones(6) * 10.0

   # Null-space gains (keeps robot away from joint limits)
   controller.null_kp = np.ones(7) * 10.0
   controller.null_kd = np.ones(7) * 1.0

   controller.set_freq(50)

   # Create desired pose (4x4 homogeneous transform)
   desired_ee = np.eye(4)
   desired_ee[:3, :3] = rotation_matrix  # 3x3 rotation
   desired_ee[:3, 3] = [x, y, z]         # position

   controller.set("ee_desired", desired_ee)  # or await in async mode

End-Effector Pose Format
~~~~~~~~~~~~~~~~~~~~~~~~~

The end-effector pose is a 4x4 homogeneous transformation matrix:

.. code-block:: python

   ee = [[R | p],
         [0 | 1]]

   # R: 3x3 rotation matrix (SO(3))
   # p: 3x1 position vector [x, y, z] in meters

Example with scipy:

.. code-block:: python

   from scipy.spatial.transform import Rotation as R

   ee = np.eye(4)
   ee[:3, :3] = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
   ee[:3, 3] = [0.5, 0.0, 0.4]  # meters

**Best for**: Cartesian motions, end-effector tracking, teleoperation.

**Default gains**: ``ee_kp = 100``, ``ee_kd = 4`` (all 6 axes); ``null_kp = 1``, ``null_kd = 1`` (per joint).


Direct Torque Control
---------------------

Send raw torque commands directly. You are responsible for computing the full torque vector.

Usage
~~~~~

.. code-block:: python

   controller.switch("torque")

   # Your custom control law
   state = controller.state
   q = state['qpos']
   dq = state['qvel']

   kp = np.ones(7) * 60.0
   kd = np.ones(7) * 3.0
   target = controller.initial_qpos

   tau = kp * (target - q) - kd * dq
   controller.torque = tau

.. warning::
   Direct torque control bypasses all built-in safety checks except torque rate limiting.
   Test in simulation first!

**Best for**: Custom control laws, gravity compensation, research.


Switching Controllers
---------------------

You can switch between controllers at runtime:

.. code-block:: python

   # Start with impedance
   controller.switch("impedance")
   controller.kp = np.ones(7) * 80.0
   controller.set("q_desired", target1)

   # Switch to OSC
   controller.switch("osc")
   controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
   controller.set("ee_desired", target2)

   # Switch to PID
   controller.switch("pid")
   controller.ki = np.ones(7) * 0.5

   # Switch to direct torque
   controller.switch("torque")
   controller.torque = np.zeros(7)

.. note::
   Switching resets initial states (``initial_qpos``, ``initial_ee``), clears rate-limiting timing, and resets the PID integral term.


Trajectory Motion
-----------------

The ``move()`` method generates a smooth, time-optimal, jerk-limited trajectory using Ruckig and executes it automatically:

.. code-block:: python

   # Move to home position
   controller.move()

   # Move to custom position
   controller.move([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

In async mode, use ``await controller.move(...)``.

``move()`` temporarily switches to impedance control. Trajectory limits are:

- Max velocity: 10 rad/s per joint
- Max acceleration: 5 rad/s per joint squared
- Max jerk: 1 rad/s per joint cubed


Safety Features
---------------

Torque Rate Limiting
~~~~~~~~~~~~~~~~~~~~

By default (``controller.clip = True``), torque commands are rate-limited to prevent safety triggers:

.. code-block:: python

   controller.clip = True                # Enable rate limiting (default)
   controller.torque_diff_limit = 990.0  # Max torque rate [Nm/s]
   controller.torque_limit = np.array([87, 87, 87, 87, 12, 12, 12])  # Absolute limits [Nm]

Gain Tuning Tips
~~~~~~~~~~~~~~~~

- Start with low gains and increase gradually
- Higher ``kp`` = stiffer tracking, but can cause oscillation
- Higher ``kd`` = more damping, reduces oscillation but slows response
- For OSC, position gains (first 3) and orientation gains (last 3) can be tuned independently
- Always test new gains with small motions first
