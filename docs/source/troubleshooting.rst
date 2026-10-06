Troubleshooting
===============

This page covers common issues and their solutions.

Connection Issues
-----------------

Error: "Robot is not ready"
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

``controller.start()`` prints a status summary and exits:

.. code-block:: text

   Joints ........... locked
   FCI .............. inactive

   Robot is not ready. Unlock first:

**Solution:**

Unlock the robot before starting the controller:

.. code-block:: python

   import aiofranka

   aiofranka.unlock()        # unlock joints + activate FCI
   controller.start()        # now this works

Or from the CLI:

.. code-block:: bash

   aiofranka unlock

Cannot Connect to Robot
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

.. code-block:: text

   ConnectionError: Unable to connect to robot

**Diagnosis:**

.. code-block:: bash

   # Test network connection
   ping 172.16.0.2

   # Should show < 1ms latency

**Solutions:**

1. **Check physical connection** — Ethernet cable connected? LED on robot's network port lit?
2. **Check IP configuration** — Correct IP? Subnet mask correct (typically 255.255.255.0)?
3. **Check firewall**

   .. code-block:: bash

      sudo ufw allow from 172.16.0.0/24

4. **Check robot state** — Robot powered on? No error lights? Can you access Franka Desk GUI in browser?

Server Not Responding
~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

.. code-block:: text

   ConnectionError: Server not responding (timeout). Check if server is running.

**Solutions:**

1. Check if the server subprocess is still alive:

   .. code-block:: python

      print(controller.running)  # True if server is responsive

2. Check server logs:

   .. code-block:: bash

      aiofranka log -n 50

3. The server may have crashed due to a control loop error. Check for error messages:

   .. code-block:: python

      # This will raise RuntimeError with the error message if server errored
      state = controller.state

Robot Behavior Issues
---------------------

Robot Triggers Safety Stop
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Robot suddenly stops moving, yellow lights flash.

**Common Causes:**

1. **Torque Rate Too High** — Lower gains or ensure smooth commands:

   .. code-block:: python

      controller.kp = np.ones(7) * 40.0   # Lower stiffness
      controller.kd = np.ones(7) * 6.0    # More damping

2. **Discontinuous Commands** — Use smooth trajectories:

   .. code-block:: python

      # Bad: jump to far target
      controller.set("q_desired", far_target)

      # Good: use trajectory generation
      controller.move(far_target)

3. **Collision Detected** — Clear workspace, check for obstacles

4. **Communication Constraints Violation** — Torque commands reached the robot late, e.g. a blocking call starved the legacy ``FrankaController``'s loop. See `Communication Constraints Violation`_ below.

**Recovery:**

.. code-block:: bash

   # Recover safety errors and re-unlock
   aiofranka unlock

Jerky or Oscillating Motion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes & Solutions:**

1. **No Rate Limiting:**

   .. code-block:: python

      # Always set frequency before control loop
      controller.set_freq(50)

2. **Low Damping:**

   .. code-block:: python

      controller.kd = np.ones(7) * 6.0

3. **High Gains:**

   .. code-block:: python

      controller.kp = np.ones(7) * 40.0  # Try lower

4. **Discontinuous Targets** — Use smooth functions:

   .. code-block:: python

      delta = np.sin(cnt / 50.0 * np.pi) * 0.1

Robot Doesn't Move
~~~~~~~~~~~~~~~~~~

**Diagnosis:**

.. code-block:: python

   print(f"Running: {controller.running}")
   state = controller.state
   print(f"Current position: {state['qpos']}")
   print(f"Desired position: {controller.q_desired}")
   diff = np.linalg.norm(controller.q_desired - state['qpos'])
   print(f"Distance to target: {diff}")

**Common Issues:**

1. **Forgot to call start()** — ``controller.start()`` must be called first
2. **Wrong controller type** — Sending ``q_desired`` but in OSC mode? Use ``controller.switch("impedance")``
3. **Target equals current** — Check the distance to target
4. **Gains too low** — Increase ``kp``

Control Loop Issues
-------------------

Communication Constraints Violation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Robot aborts motion with ``communication_constraints_violation`` error.

**Cause:** Torque commands reached the robot late. With the legacy Python loop
(``FrankaController``, ``aiofranka start-server --python``), a blocking call in your code
starved it. The native loop does not wait for Python, so with it the cause is the system: a
saturated CPU, the network, or a loop thread without real-time priority.

**Solutions:**

- Use the native loop: ``NativeFrankaController`` in your process, or server mode
  (``FrankaRemoteController``), whose server runs it by default
- With the legacy ``FrankaController``, follow the :doc:`async_mode` guide to avoid blocking the
  event loop
- With the native loop, use wired Ethernet, keep a core free for the robot NIC's interrupts, and
  give the loop's thread its SCHED_FIFO priority (see :doc:`native`). ``aiofranka rt-benchmark``
  measures the loop (see :doc:`cli`).

Low Control Frequency
~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** ``test_connection()`` shows frequency < 990 Hz or high jitter.

**Solutions:**

- Use a wired Ethernet connection (not WiFi)
- Reduce system load (close unnecessary programs)
- Consider a real-time Linux kernel
- Move heavy computation to a separate process (server mode handles this automatically)

OSC Issues
----------

OSC Unstable or Oscillates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:**

1. **Near singularity:**

   .. code-block:: python

      jac = controller.state['jac']
      cond = np.linalg.cond(jac)
      if cond > 100:
          print(f"Warning: Poor conditioning: {cond}")

2. **Gains too high:**

   .. code-block:: python

      controller.ee_kp = np.array([200, 200, 200, 600, 600, 600])
      controller.ee_kd = np.ones(6) * 8.0

3. **Null-space conflict:**

   .. code-block:: python

      controller.null_kp = np.ones(7) * 5.0
      controller.null_kd = np.ones(7) * 1.0

OSC Doesn't Reach Target
~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnosis:**

.. code-block:: python

   state = controller.state
   ee_current = state['ee']
   ee_desired = controller.ee_desired

   pos_error = np.linalg.norm(ee_desired[:3, 3] - ee_current[:3, 3])
   print(f"Position error: {pos_error * 1000:.1f} mm")

**Causes:**

1. Target unreachable (outside workspace)
2. Null-space pulling away — adjust ``null_kp`` / ``null_kd``
3. Singularity — move via intermediate poses

Programming Issues
------------------

TypeError: 'coroutine' object is not iterable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Forgot to ``await`` an async function.

.. code-block:: python

   # Wrong (async mode)
   controller.start()

   # Correct (async mode)
   await controller.start()

.. note::
   This only applies to async mode. In server mode, ``controller.start()`` is synchronous.

RuntimeError: This event loop is already running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Using ``asyncio.run()`` inside an async function.

.. code-block:: python

   # Wrong
   async def my_function():
       asyncio.run(controller.start())

   # Correct
   async def my_function():
       await controller.start()

Getting Help
------------

Before asking for help:

1. Check this guide — is your issue covered?
2. Test in simulation — does it work with ``RobotInterface(None)``?
3. Review the examples in :doc:`examples`
4. Check server logs: ``aiofranka log -n 50``
5. Simplify — does a minimal example reproduce the issue?

When asking for help, include:

1. **aiofranka version**: ``pip show aiofranka``
2. **Operating system and Python version**
3. **Minimal code** that reproduces the issue
4. **Full error message** from terminal
5. **Whether you're using async mode or server mode**, with the native or the legacy Python loop

Where to ask:

- **GitHub Issues**: https://github.com/younghyopark/aiofranka/issues
