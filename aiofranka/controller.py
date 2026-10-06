import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import sys 
sys.path.append(".")
import threading
import asyncio
import time
import logging
from tqdm import trange
from pathlib import Path
import numpy as np 
import time 
from aiofranka.robot import RobotInterface, set_macos_control_thread_qos
from aiofranka.payload import identify_payload
from aiofranka.config import load_config, tcp_transform
from ruckig import InputParameter, Ruckig, Trajectory, Result

logger = logging.getLogger(__name__)

CUR_DIR = Path(__file__).parent.resolve()



class FrankaController:
    """
    High-level asyncio controller for Franka robots with multiple control modes.

    Legacy: its 1 kHz loop runs in Python, on the event loop, where anything else that
    runs (a planner, garbage collection, a thread holding the GIL) delays the next
    torque command. Use NativeFrankaController, which has this API and runs the loop in
    C++; FrankaController keeps working for existing code, and for a subclass's step().

    This controller runs a 1kHz torque control loop in the background using asyncio,
    while allowing you to send high-level commands asynchronously. Supports three
    control modes:
    
    1. Impedance Control: Joint-space spring-damper control
    2. Operational Space Control (OSC): Task-space control with null-space
    3. Direct Torque Control: Raw torque commands
    
    The controller automatically handles:
    - Background control loop at 1kHz
    - Torque rate limiting for safety
    - Thread-safe state updates
    - Rate-limited command updates
    - Smooth trajectory generation
    
    Attributes:
        robot (RobotInterface): Robot interface instance
        type (str): Current controller type ("impedance", "osc", "torque")
        running (bool): Whether control loop is active
        state (dict): Current robot state (updated at 1kHz)
        
        # Impedance control gains
        kp (np.ndarray): Joint position stiffness [Nm/rad] (7,)
        kd (np.ndarray): Joint damping [Nm⋅s/rad] (7,)
        
        # OSC gains
        ee_kp (np.ndarray): EE stiffness as an acceleration gain [1/s²] (6,): the OSC
            commands the wrench Λ·(ee_kp·e − ee_kd·v), with Λ the task-space inertia, so
            ee_kp = ω² and ee_kd = 2ζω for a natural frequency ω and damping ratio ζ
        ee_kd (np.ndarray): EE damping as an acceleration gain [1/s] (6,)
        null_kp (np.ndarray): Null-space stiffness [Nm/rad] (7,)
        null_kd (np.ndarray): Null-space damping [Nm⋅s/rad] (7,)
        
        # Target states
        q_desired (np.ndarray): Desired joint positions [rad] (7,)
        ee_desired (np.ndarray): Desired pose of the OSC control frame as 4x4 transform
        torque (np.ndarray): Direct torque command [Nm] (7,)
        control_transform (np.ndarray): Tool center point that the OSC controls, in
            the flange frame, 4x4 (default: identity, i.e. the flange); see set_tcp()
        last_command (np.ndarray): Last torque sent to the robot, after rate
            limiting and clipping [Nm] (7,)

        # Safety
        clip (bool): Enable torque rate limiting (default: True)
        torque_diff_limit (float): Max torque rate [Nm/s] (default: 990)
        
    Examples:
        Basic usage:
            >>> robot = RobotInterface("172.16.0.2")
            >>> controller = FrankaController(robot)
            >>> await controller.start()
            >>> controller.switch("impedance")
            >>> controller.set_freq(50)
            >>> await controller.set("q_desired", target_joints)
            >>> await controller.stop()
        
        OSC control:
            >>> controller.switch("osc")
            >>> controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
            >>> controller.set_freq(50)
            >>> desired_ee = np.eye(4)
            >>> desired_ee[:3, 3] = [0.4, 0.0, 0.5]
            >>> await controller.set("ee_desired", desired_ee)
        
        Direct torque:
            >>> controller.switch("torque")
            >>> controller.torque = np.zeros(7)  # Zero torques
        
    Caveats:
        - Must await controller.start() before sending commands
        - Use set_freq() before set() to enforce timing
        - High gains can cause instability or safety triggers
        - Switching controllers resets initial state
        - State access is thread-safe but copy if modifying
        - Control loop must run continuously at ~1kHz
    """

    def __init__(self, robot: RobotInterface):
        """
        Initialize controller with robot interface.
        
        Args:
            robot (RobotInterface): Initialized robot interface
            
        Note:
            Controller is initialized in "impedance" mode with conservative gains.
            Call start() to begin the control loop, then switch() to change modes.
            
        Default Gains:
            - Impedance: kp=80, kd=4 (per joint)
            - OSC: ee_kp=[100]*6, ee_kd=[4]*6
            - Null-space: null_kp=1, null_kd=1
        """
        self.robot = robot
        self.control_transform = np.eye(4)
        self.last_command = np.zeros(7)

        self.initialize()
        self.state_lock = threading.Lock()

        self.type = "impedance"
        self.running = False
        self.task = None
        self.clip = True
        self.error_callback = None  # Callback function(error_str) called on control loop exception


        self.kp, self.kd = np.ones(7) * 80, np.ones(7) * 4
        self.ki = np.ones(7) * 0.1  # Integral gains
        self.error_integral = np.zeros(7)  # Accumulated error
        self.ee_kp, self.ee_kd = np.ones(6) * 100, np.ones(6) * 4
        self.null_kp, self.null_kd = np.ones(7) * 1, np.ones(7) * 1


        self.track = False 
        self.torque_diff_limit = 990.
        self.torque_limit = np.array([87, 87, 87, 87, 12, 12, 12])  # Nm
        
        # Rate limiting for .set() method
        self._update_freq = 50.0  # Default 50Hz
        self._last_update_time = {}
        self._pending_updates = {}

        self.state = None 

        # move() is done, and activate() takes an OSC configuration's null-space target,
        # when every joint is this close to its target [rad].
        self.arrival_tolerance = 0.03

        self.verbose = False


    async def test_connection(self): 
        """
        Test control loop timing and diagnose connection quality.
        
        Runs for 5 seconds and prints statistics about control loop performance:
        - Actual frequency (should be ~1000 Hz)
        - Mean/std/min/max loop time
        - Jitter (max - min)
        
        Use this to verify your setup is working correctly before running experiments.
        
        Example Output:
            Control loop stats (last 1000 iterations):
              Frequency: 1000.2 Hz (target: 1000 Hz)
              Mean dt: 1.000 ms, Std: 0.015 ms
              Min dt: 0.985 ms, Max dt: 1.025 ms
              Jitter (max-min): 0.040 ms
              
        Caveats:
            - High jitter (>0.5ms) indicates system load or network issues
            - Frequency <990 Hz suggests performance problems
            - Run on a dedicated realtime system for best results
        """

        self.track = True 
        await asyncio.sleep(5)
        self.track = False

    def initialize(self):

        # Read from already-synced MuJoCo state to avoid calling readOnce()
        # concurrently with the active torque control loop (would drop TCP connection)
        self.initial_qpos = deepcopy(self.robot.data.qpos)
        self.initial_qvel = deepcopy(self.robot.data.qvel)
        self.initial_ee = self.robot._ee() @ self.control_transform

        self.q_desired = self.initial_qpos
        self.ee_desired = self.initial_ee

    def _update_desired(self, desired):
        """
        Update the desired joint positions, kp and kd
        This function is called by the server when a client sends new values
        Thread-safe update of shared state.
        
        Args:
            desired: Desired joint positions (7-element array)
            kp: Joint position stiffness gains (7-element array)
            kd: Joint velocity damping gains (7-element array)
        """
        with self.state_lock:
            self.q_desired = np.array(desired) if type(desired) == list else desired
    
    def set_freq(self, freq: float):
        """
        Set the update frequency for rate-limited set() calls.
        
        This enforces strict timing for subsequent set() calls, automatically
        sleeping to maintain the specified frequency. Prevents sending commands
        too fast and ensures smooth, consistent control.
        
        Args:
            freq (float): Desired update frequency in Hz (typically 10-100 Hz)
            
        Note:
            Must be called BEFORE the first set() call to take effect.
            Each attribute tracked by set() has independent timing.
            
        Examples:
            >>> controller.set_freq(50)  # 50 Hz updates
            >>> for i in range(100):
            ...     await controller.set("q_desired", compute_target())
            ...     # Automatically sleeps to maintain 50 Hz
            
        Caveats:
            - Don't set freq > 200 Hz (unnecessary and may cause timing issues)
            - Lower freq = smoother motion but slower response
            - Higher freq = faster response but requires more computation
            - Timing is per-attribute (q_desired and ee_desired tracked separately)
        """
        self._update_freq = freq
    
    async def set(self, attr: str, value):
        """
        Rate-limited setter that enforces strict timing for control updates.
        
        This method ensures updates to controller attributes happen at the
        frequency specified by set_freq(). It compensates for drift by tracking
        the target time for each update, guaranteeing consistent timing even
        if your computation time varies.
        
        Args:
            attr (str): Attribute name to set. Common values:
                       - "q_desired": Joint position target (impedance mode)
                       - "ee_desired": End-effector pose target (OSC mode)
                       - "torque": Direct torque command (torque mode)
            value: Value to set. Type depends on attr:
                  - q_desired: np.ndarray (7,) [rad]
                  - ee_desired: np.ndarray (4, 4) homogeneous transform
                  - torque: np.ndarray (7,) [Nm]
                  
        Note:
            - Automatically sleeps to maintain frequency set by set_freq()
            - First call for an attribute initializes timing
            - Each attribute has independent timing tracking
            - Thread-safe update of shared state
            
        Examples:
            Impedance control:
                >>> controller.set_freq(50)
                >>> for i in range(100):
                ...     target = initial_q + np.sin(i / 50.0 * np.pi) * 0.1
                ...     await controller.set("q_desired", target)
            
            OSC control:
                >>> controller.set_freq(100)
                >>> desired_ee = np.eye(4)
                >>> desired_ee[:3, 3] = [0.5, 0.0, 0.3]
                >>> await controller.set("ee_desired", desired_ee)
            
        Caveats:
            - Must call set_freq() before first set() call
            - If computation takes longer than 1/freq, timing will slip
            - Don't mix set() and direct attribute assignment
            - Don't call set() faster than the specified frequency
        """
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq

        # Initialize tracking for this attribute if first time
        if attr not in self._last_update_time:
            self._last_update_time[attr] = current_time
            with self.state_lock:
                setattr(self, attr, value)
            await asyncio.sleep(dt)
            self._last_update_time[attr] = current_time + dt
            return

        # Calculate target time for this update
        target_time = self._last_update_time[attr] + dt

        # Set value immediately to minimize latency
        with self.state_lock:
            setattr(self, attr, value)

        # Then sleep to rate-limit the next call
        sleep_time = target_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        # Update last update time to target (not actual) to avoid drift
        self._last_update_time[attr] = target_time


    async def _run(self): 
        """Run the control loop continuously in the background"""
        # On macOS, libfranka busy-waits for robot states, which needs this thread on a
        # performance core. It is not always the thread that created the Robot.
        set_macos_control_thread_qos()
        self.running = True
        
        loop_times = []
        last_time = time.perf_counter()
        log_interval = 1000  # Log every 1000 iterations (1 second at 1000Hz)
        iteration = 0
        
        try:
            while self.running: 

                
                t0 = time.time() 
                self.step()
                
                # Track timing
                if self.track:
                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    loop_times.append(dt)
                    last_time = current_time
                    iteration += 1
                    
                    # Log statistics every log_interval iterations
                    if iteration % log_interval == 0:
                        loop_times_array = np.array(loop_times)
                        mean_dt = np.mean(loop_times_array) * 1000  # Convert to ms
                        std_dt = np.std(loop_times_array) * 1000
                        min_dt = np.min(loop_times_array) * 1000
                        max_dt = np.max(loop_times_array) * 1000
                        actual_freq = 1.0 / np.mean(loop_times_array)
                        
                        print(f"Control loop stats (last {log_interval} iterations):")
                        print(f"  Frequency: {actual_freq:.1f} Hz (target: 1000 Hz)")
                        print(f"  Mean dt: {mean_dt:.3f} ms, Std: {std_dt:.3f} ms")
                        print(f"  Min dt: {min_dt:.3f} ms, Max dt: {max_dt:.3f} ms")
                        print(f"  Jitter (max-min): {max_dt - min_dt:.3f} ms")
                        
                        loop_times.clear()

                dt = time.time() - t0
                if not self.robot.real: 
                    await asyncio.sleep(1/1000. - dt)  # Yield control to event loop
                else:
                    await asyncio.sleep(0)  # Yield control to event loop
        except Exception as e:
            self.running = False
            error_str = str(e)
            print(f"Error in control loop: {error_str}")
            # Call error callback if set
            if self.error_callback is not None:
                try:
                    self.error_callback(error_str)
                except Exception as cb_err:
                    print(f"Error in error_callback: {cb_err}")
            sys.exit(1)  # Kill the entire script
    
    async def start(self):
        """
        Start the 1kHz background control loop.
        
        This creates an asyncio task that runs the control loop continuously
        at ~1000 Hz. The loop reads robot state, computes control torques based
        on the current controller type, and sends torque commands.
        
        Returns:
            asyncio.Task: The background control loop task
            
        Note:
            - Blocks for 1 second to ensure loop starts successfully
            - Starts robot torque control mode automatically
            - Control loop runs until stop() is called
            - You can send commands via set() while loop is running
            
        Example:
            >>> await controller.start()
            >>> # Control loop now running in background
            >>> controller.set_freq(50)
            >>> await controller.set("q_desired", target)
            >>> await controller.stop()
            
        Caveats:
            - Must be awaited (async function)
            - Don't call start() multiple times without stop()
            - If loop crashes, robot will trigger safety stop
            - Check terminal for error messages if robot stops unexpectedly
        """

        logger.info("Starting robot control loop")
        self.robot.start()

        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(1)  # Yield to ensure the task starts
        return self.task
    
    async def stop(self):
        """
        Stop the background control loop and robot.
        
        Gracefully terminates the control loop task and stops robot torque control.
        Robot will hold position briefly then release brakes.
        
        Note:
            - Blocks for 1 second to ensure clean shutdown
            - Cancels asyncio control loop task
            - Stops robot torque control mode
            - Safe to call multiple times
            
        Example:
            >>> await controller.start()
            >>> # ... do control ...
            >>> await controller.stop()
            
        Caveat:
            Ensure robot is in a safe configuration before stopping. Robot
            will briefly hold position then release brakes.
        """
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                print("Control loop task cancelled.")

        self.robot.stop()
        print("robot stopped")
        await asyncio.sleep(1)  # Yield to ensure the task starts

    def switch(self, controller_type: str):
        """
        Switch between control modes at runtime.

        Changes the active controller without stopping the control loop. Resets
        the initial state (initial_qpos, initial_ee) to current robot state and
        clears rate-limiting timing.

        Args:
            controller_type (str): Controller type to switch to:
                - "impedance": Joint-space impedance control
                - "pid": Joint-space PID control with integral term
                - "osc": Operational space control (task space)
                - "torque": Direct torque control

        Example:
            >>> controller.switch("impedance")
            >>> controller.kp = np.ones(7) * 80
            >>> # ... run impedance control ...
            >>>
            >>> controller.switch("osc")
            >>> controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
            >>> # ... run OSC control ...

        Note:
            - Can be called while control loop is running
            - Resets q_desired to current position
            - Resets ee_desired to current pose of the control frame
            - Clears timing state from previous set() calls
            - Resets integral term when switching to/from PID
            
        Caveats:
            - Switching causes brief discontinuity in control
            - Adjust gains after switching for smooth transition
            - Don't switch rapidly (< 1 Hz) as it resets state
        """
        self.type = controller_type
        self.initialize()
        # Reset integral term when switching controllers
        self.error_integral = np.zeros(7)
        # Reset timing state when switching controllers
        self._last_update_time.clear()

        if self.verbose:
            print("==================================")
            print(f"Switched to {controller_type} controller.")
            print("==================================")

    def set_tcp(self, transform):
        """
        Set the tool center point (TCP) that the OSC controls, relative to the flange.

        The OSC then tracks ee_desired with this frame instead of the flange. Express
        it in the flange frame (the attachment_site of the MuJoCo model): its origin
        is the center of the flange face, z points out of the flange, and the box on
        the side of link 7 points between +x and +y. The docs show it on the robot
        (Controllers > Operational Space Control > Tool Center Point). ee_desired
        becomes the current pose of the new TCP, so the arm holds still.

        The TCP of the end-effector profile in Desk is not used here.

        Args:
            transform (array-like): Pose of the TCP in the flange frame as a 4x4
                transform, or its translation [m] (3,) for the flange's orientation

        Raises:
            ValueError: If transform is not a translation or a 4x4 pose

        Example:
            >>> controller.switch("osc")
            >>> controller.set_tcp([0, 0, 0.1034])  # Franka Hand fingertips
            >>> target = controller.ee_desired.copy()
            >>> target[2, 3] -= 0.05  # 5 cm down, in the base frame
            >>> await controller.set("ee_desired", target)
        """
        transform = tcp_transform(transform)
        with self.state_lock:
            self.control_transform = transform
            # Track the new TCP where it is now, so the arm holds still.
            self.initial_ee = self.robot._ee() @ transform
            self.ee_desired = self.initial_ee.copy()

    def check_tool(self, config):
        """
        Check that the tool a configuration needs is the end-effector profile active in Desk.

        The profile is the one RobotInterface read from Desk when it connected. A
        configuration without a tool, and MuJoCo, which has no Desk, pass.

        Args:
            config (str | Path | dict): Controller configuration (see aiofranka.config)

        Raises:
            RuntimeError: If another tool is active, or Desk could not be read
        """
        config = load_config(config)
        name = config.get("tool")
        if name is None or not self.robot.real:
            return
        active = self.robot.tool
        if active is None:
            raise RuntimeError(f"The configuration needs the tool {name!r}, but {self.robot.tool_error}. "
                               "Pass check_tool=False to skip the check.")
        from aiofranka.tools import NO_END_EFFECTOR
        matches = active.id == NO_END_EFFECTOR if name.lower() == "none" else active.name == name
        if matches:
            return
        fix = "aiofranka tool unload" if name.lower() == "none" else f"aiofranka tool load '{name}'"
        raise RuntimeError(f"The configuration needs the tool {name!r}, but the end effector active in Desk "
                           f"is {active.name!r}. Activate it with: {fix} (then reconnect)")

    def activate(self, config, check_tool=True, check_null_target=True):
        """
        Apply a controller configuration: its mode, gains and policy rate.

        For impedance it sets kp and kd; for osc the TCP, ee_kp, ee_kd, null_kp,
        null_kd and the null-space target. It then sets the rate of set() to the
        configuration's frequency. Like switch(), it holds the current joint positions
        (impedance) or TCP pose (osc). See aiofranka.config for the file format.

        An OSC configuration without a null_target keeps the joint positions at
        activation as the null-space target, as switch() does. One with a null_target
        (e.g. a policy's) is only activated with the arm already there, every joint
        within arrival_tolerance, the criterion move() stops at; otherwise the null
        space would swing the arm toward it at once. Move there first:

            >>> config = aiofranka.load_config("configs/pocky/lv1_osc.yaml")
            >>> await controller.move(config["null_target"])
            >>> controller.activate(config)

        Reading a file takes a few milliseconds, which delays the 1 kHz loop that
        long; while it runs, pass a configuration read before with
        aiofranka.load_config().

        Args:
            config (str | Path | dict): YAML file, or its contents
            check_tool (bool): Refuse a configuration whose tool is not the end
                effector active in Desk (see check_tool())
            check_null_target (bool): Refuse an OSC configuration whose null_target
                the arm is not at. Only skip it where the null space is known to be at
                rest, e.g. at a pose where the TCP Jacobian's null space holds no
                error toward the target.

        Returns:
            dict: The configuration, as aiofranka.load_config() reads it

        Raises:
            ValueError: If the configuration is not valid
            RuntimeError: If its tool is not the one active in Desk, or the arm is
                not at its null_target

        Example:
            >>> controller.activate("configs/osc.yaml")
            >>> target = controller.ee_desired.copy()
            >>> target[2, 3] -= 0.05
            >>> await controller.set("ee_desired", target)  # at the configured frequency
        """
        config = load_config(config)
        if check_tool:
            self.check_tool(config)
        if config["mode"] == "osc" and config["null_target"] is not None and check_null_target:
            error = np.abs(np.array(self.robot.data.qpos[:7]) - config["null_target"])
            if error.max() > self.arrival_tolerance:
                raise RuntimeError(
                    f"The arm is {error.max():.3f} rad from the configuration's null_target at joint "
                    f"{error.argmax() + 1} (more than arrival_tolerance, {self.arrival_tolerance} rad), so the "
                    "null space would swing it there. Move there first: "
                    "await controller.move(config['null_target'])")
        if config["mode"] == "osc":
            self.set_tcp(config["tcp"])
        self.switch(config["mode"])
        with self.state_lock:
            if config["mode"] == "impedance":
                self.kp, self.kd = config["kp"].copy(), config["kd"].copy()
            else:
                self.ee_kp, self.ee_kd = config["ee_kp"].copy(), config["ee_kd"].copy()
                self.null_kp, self.null_kd = config["null_kp"].copy(), config["null_kd"].copy()
                if config["null_target"] is not None:  # else the joint positions now, as switch() sets
                    self.initial_qpos = config["null_target"].copy()  # the OSC's null-space target
        self.set_freq(config["frequency"])
        return config

    def step(self):
        self.state = self.robot.state
        if self.type == "impedance":
            self._impedance_step(self.state)
        elif self.type == "pid":
            self._pid_step(self.state)
        elif self.type == "osc":
            self._osc_step(self.state)
        elif self.type == "torque":
            self._torque_step(self.state)
        else:
            raise ValueError(f"Unknown controller type: {self.type}")

    def _send(self, torque):
        self.last_command = torque
        self.robot.step(torque)

    def _torque_step(self, state):

        self._send(self.torque)

    def _osc_step(self, state):

        # Pose and Jacobian of the control frame. Its origin moves with
        # v + w x r, where r is its offset from attachment_site.
        ee = state['ee'] @ self.control_transform
        offset = ee[:3, 3] - state['ee'][:3, 3]
        jac = state['jac'].copy()
        jac[:3] += np.cross(jac[3:].T, offset).T
        q = state['qpos']
        dq = state['qvel']
        mm = state['mm']
        last_torque = state['last_torque']


        with self.state_lock:
            ee_goal = self.ee_desired

        position_error = ee_goal[:3, 3] - ee[:3, 3]

        rotation_error = R.from_matrix(ee_goal[:3, :3]) * R.from_matrix(ee[:3, :3]).inv()
        rotation_error_vec = rotation_error.as_rotvec()

        twist = np.zeros(6)
        twist[:3] = position_error
        twist[3:] = rotation_error_vec
        ee_vel = jac @ dq
        # minv = np.linalg.inv(mm)
        if abs(np.linalg.det(mm)) > 1e-2:
            minv = np.linalg.inv(mm)
        else:
            minv = np.linalg.pinv(mm)
        mx_inv = jac @ minv @ jac.T

        if abs(np.linalg.det(mx_inv)) > 1e-2:
            mx = np.linalg.inv(mx_inv)
        else:
            mx = np.linalg.pinv(mx_inv)

        # operational space feedback torque
        feedback = jac.T @ mx @ (self.ee_kp * twist - self.ee_kd * ee_vel)

        # null space torque
        q0 = self.initial_qpos
        ddq = self.null_kp * (q0 - q) - self.null_kd * dq
        jbar = minv @ jac.T @ mx
        null = (np.eye(7) - jac.T @ jbar.T) @ ddq

        # Add coriolis compensation
        tau_d = feedback + null

        # make sure torque rate of change is not too high
        if self.clip:
            diff = (tau_d - last_torque)/1e-3
            diff = np.clip(diff, -self.torque_diff_limit, self.torque_diff_limit)
            tau_d = last_torque + diff * 1e-3

            tau_d = np.clip(tau_d, -self.torque_limit, self.torque_limit)

        self._send(tau_d)




    def _pid_step(self, robot_state):
        """
        PID control in joint space with integral term for steady-state error.
        
        Computes: τ = Kp*e + Ki*∫e*dt - Kd*dq
        where e = q_desired - q
        """
        # Get state variables
        q = np.array(robot_state['qpos'])
        dq = np.array(robot_state['qvel'])
        last_torque = robot_state['last_torque']

        # Get current target (thread-safe)
        with self.state_lock:
            kp = self.kp
            ki = self.ki
            kd = self.kd
            q_desired = self.q_desired
            q_goal = q_desired
    
        position_error = q_goal - q
        
        # Update integral term (dt = 1ms = 0.001s)
        self.error_integral += position_error * 1e-3
        
        # Anti-windup: clamp integral term
        integral_limit = 10.0  # Nm*s (adjust as needed)
        self.error_integral = np.clip(self.error_integral, -integral_limit, integral_limit)

        # Compute PID control
        tau = position_error * kp + self.error_integral * ki - dq * kd

        tau_d = tau
        
        # Torque rate limiting
        if self.clip:
            diff = (tau_d - last_torque)/1e-3
            diff = np.clip(diff, -self.torque_diff_limit, self.torque_diff_limit)
            tau_d = last_torque + diff * 1e-3

        self.torque = tau_d
        self._send(tau_d)

    def _impedance_step(self, robot_state): 


        # Get state variables
        q = np.array(robot_state['qpos'])
        dq = np.array(robot_state['qvel'])
        last_torque = robot_state['last_torque']

        # Get current target from trajectory (thread-safe)
        with self.state_lock:
            kp = self.kp
            kd = self.kd
            q_desired = self.q_desired
            q_goal = q_desired
    
        position_error = q_goal - q

        # Compute joint-space impedance control
        tau = position_error * kp - dq * kd

        # Add coriolis compensation
        tau_d = tau #+ coriolis
        
        # make sure torque rate of change is not too high
        if self.clip:
            diff = (tau_d - last_torque)/1e-3
            diff = np.clip(diff, -self.torque_diff_limit, self.torque_diff_limit)
            tau_d = last_torque + diff * 1e-3

            tau_d = np.clip(tau_d, -self.torque_limit, self.torque_limit)
            
        self.torque = tau_d

        self._send(tau_d)


    async def move(self, qpos = [0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853]):
        """
        Move robot to target joint position using smooth trajectory.
        
        Generates a time-optimal, jerk-limited trajectory using Ruckig online
        trajectory generation, then executes it at 50 Hz. Automatically switches
        to impedance mode if not already active.
        
        Args:
            qpos (list | np.ndarray): Target joint positions [rad] (7,)
                                     Default: Home position
                                     
        Note:
            - Uses Ruckig for smooth, time-optimal trajectories
            - Respects velocity, acceleration, and jerk limits
            - Switches to impedance control automatically
            - Executes trajectory at 50 Hz (20ms updates)
            - Duration depends on distance and limits
            
        Trajectory Limits:
            - Max velocity: 10 rad/s per joint
            - Max acceleration: 5 rad/s² per joint
            - Max jerk: 1 rad/s³ per joint
            
        Examples:
            Move to home position:
                >>> await controller.move()
            
            Move to custom position:
                >>> target = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
                >>> await controller.move(target)
            
            Move to current position + offset:
                >>> current = controller.state['qpos']
                >>> await controller.move(current + np.array([0.1, 0, 0, 0, 0, 0, 0]))
                
        Caveats:
            - Large motions take longer (trajectory is time-optimal)
            - Don't call while other control is active
            - Blocks until motion completes: until every joint is within
              arrival_tolerance (0.03 rad by default) of the target, or 3 s after the
              trajectory if friction keeps it farther (it then prints how far)
            - Switches to impedance mode (resets controller state)
            - May fail if target is at joint limits or in collision
        """
        self.type = "impedance"
        target = np.array(qpos, dtype=float)

        inp = InputParameter(7)
        # The state the control loop synced last; reading the robot here would take a
        # state from the 1 kHz loop.
        inp.current_position = np.array(self.robot.data.qpos[:7])
        inp.current_velocity = np.array(self.robot.data.qvel[:7])
        inp.current_acceleration = np.zeros(7)

        inp.target_position = target
        inp.target_velocity = np.zeros(7)
        inp.target_acceleration = np.zeros(7)

        inp.max_velocity = np.ones(7) * 10
        inp.max_acceleration = np.ones(7) * 5
        inp.max_jerk = np.ones(7)
        
        otg = Ruckig(7)
        trajectory = Trajectory(7)

        otg.calculate(inp, trajectory)

        # create a trajectory to the desired qpos  (linear interpolation)
        n_steps = int(trajectory.duration * 50)
        eta_total = trajectory.duration
        # The trajectory is sampled at 50 Hz, so play it at 50 Hz whatever set_freq() says.
        update_freq, self._update_freq = self._update_freq, 50.0
        self._last_update_time.pop("q_desired", None)
        try:
            for i in range(n_steps):
                q_desired, _, _ = trajectory.at_time(i / 50.0)
                await self.set("q_desired", q_desired)
                done = (i + 1) * 20 // n_steps
                bar = "█" * done + "░" * (20 - done)
                eta = eta_total - (i + 1) / 50.0
                print(f"\r  Moving [{bar}] {(i + 1) * 100 // n_steps}% ETA {eta:.1f}s", end="", flush=True)
        finally:
            self._update_freq = update_freq
            self._last_update_time.pop("q_desired", None)
        print()

        # Then hold the exact target until every joint is within arrival_tolerance of it.
        with self.state_lock:
            self.q_desired = target.copy()
        deadline = time.perf_counter() + 3.0
        while (error := np.abs(np.array(self.robot.data.qpos[:7]) - target).max()) > self.arrival_tolerance:
            if time.perf_counter() > deadline:
                print(f"  move(): still {error:.3f} rad from the target, more than arrival_tolerance "
                      f"({self.arrival_tolerance} rad); a stiffer kp gets closer")
                break
            await asyncio.sleep(0.01)

    async def identify_payload(self, tool_length=0.2, tool_radius=0.1, floor=0.0, **kwargs):
        """
        Identify the mass and center of mass of the tool on the flange.

        Moves through poses around the current one, measures the joint torques at
        rest that the robot does not compensate, and fits the tool to them. It only
        measures: save the result with aiofranka.save_tool() and activate it with
        aiofranka.load_tool(). See aiofranka.payload.identify_payload() for the
        details and further arguments.

        Args:
            tool_length (float): Length of the tool from the flange [m], for the
                collision checks (default: 0.2)
            tool_radius (float): Radius of the tool [m] (default: 0.1)
            floor (float): Height of the floor or table in the base frame [m]
                (default: 0.0)
            **kwargs: Further arguments of aiofranka.payload.identify_payload()

        Returns:
            PayloadEstimate: Estimated tool, with standard errors

        Example:
            >>> estimate = await controller.identify_payload(tool_length=0.25)
            >>> aiofranka.save_tool("gripper", estimate.mass, estimate.com)
            >>> aiofranka.load_tool("gripper")
        """
        return await identify_payload(
            self, tool_length=tool_length, tool_radius=tool_radius, floor=floor, **kwargs,
        )
