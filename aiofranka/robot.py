import os
import sys
import mujoco
import mujoco.viewer
from pathlib import Path
import numpy as np
import time
import requests
from aiofranka.client import FrankaLockUnlock

CUR_DIR = Path(__file__).parent.resolve()


def set_macos_control_thread_qos():
    """
    Give the calling thread the scheduling that libfranka needs on macOS.

    On macOS, libfranka busy-waits for robot states, which only keeps up with the
    1 kHz control loop on a performance core. libfranka sets the QoS class
    USER_INTERACTIVE for the thread that creates the Robot; call this at the start
    of the thread that runs the control loop. Does nothing on other platforms or
    when busy-waiting is disabled with LIBFRANKA_MACOS_BUSY_WAIT=0.

    Returns:
        bool: True if the QoS class was set.
    """
    if sys.platform != "darwin" or os.environ.get("LIBFRANKA_MACOS_BUSY_WAIT") == "0":
        return False
    import ctypes

    QOS_CLASS_USER_INTERACTIVE = 0x21
    libc = ctypes.CDLL(None)
    return libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0


_MACOS_PYLIBFRANKA_MISSING = """pylibfranka is not installed.

PyPI has no official macOS build of pylibfranka. On Apple Silicon Macs with
macOS 15 or newer, install the unofficial build:

    pip install pylibfranka-macos

On other Macs, build it from source from younghyopark/libfranka, a fork with
macOS support, see https://github.com/younghyopark/aiofranka#macos-apple-silicon"""


def require_pylibfranka():
    """
    Import pylibfranka, explaining how to install it on macOS if it is missing.

    On macOS, aiofranka only depends on pylibfranka-macos, an unofficial build of
    pylibfranka, for Apple Silicon and macOS 15 or newer.

    Returns:
        module: The pylibfranka module.

    Raises:
        ModuleNotFoundError: If pylibfranka is not installed.
    """
    try:
        import pylibfranka
    except ModuleNotFoundError as e:
        if sys.platform == "darwin" and e.name == "pylibfranka":
            raise ModuleNotFoundError(_MACOS_PYLIBFRANKA_MISSING, name=e.name) from e
        raise
    return pylibfranka


def merge_payload(model, link_inertial, mass, com, inertia, site_id=None):
    """
    Merge a payload on the flange into the last link of a MuJoCo FR3 model.

    Sets fr3_link7's mass, center of mass and inertia to those of the link and the
    payload together (parallel axis theorem), as RobotInterface.sync_payload() does for
    the model aiofranka's controllers use. Call mj_setConst() afterwards.

    Args:
        model (mujoco.MjModel): Model to change
        link_inertial (tuple): fr3_link7's own mass, ipos, iquat and inertia, as in the
            unmodified model
        mass (float): Payload mass [kg]
        com (array-like): Payload center of mass in the flange frame [m] (3,)
        inertia (array-like): Payload inertia about its center of mass in the flange
            frame [kg m^2] (3, 3)
        site_id (int | None): The flange site (default: attachment_site)
    """
    def rotation(quat):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        return mat.reshape(3, 3)

    if site_id is None:
        site_id = model.site("attachment_site").id
    com, inertia = np.asarray(com, dtype=float), np.asarray(inertia, dtype=float)
    body = model.body("fr3_link7").id
    link_mass, link_com, link_iquat, link_inertia = link_inertial
    rot = rotation(link_iquat)
    link_tensor = rot @ np.diag(link_inertia) @ rot.T

    # Payload in the link frame, from the attachment_site placement.
    rot = rotation(model.site_quat[site_id])
    load_com = model.site_pos[site_id] + rot @ com
    load_tensor = rot @ inertia @ rot.T

    # Combine both about their joint center of mass (parallel axis theorem).
    total = link_mass + mass
    total_com = (link_mass * link_com + mass * load_com) / total

    def parallel_axis(m, offset):
        return m * (offset @ offset * np.eye(3) - np.outer(offset, offset))

    tensor = (
        link_tensor + parallel_axis(link_mass, link_com - total_com)
        + load_tensor + parallel_axis(mass, load_com - total_com)
    )
    principal, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, axes.flatten())

    model.body_mass[body] = total
    model.body_ipos[body] = total_com
    model.body_iquat[body] = quat
    model.body_inertia[body] = principal


def link_inertial(model):
    """fr3_link7's mass, ipos, iquat and inertia, for merge_payload()."""
    body = model.body("fr3_link7").id
    return (float(model.body_mass[body]), model.body_ipos[body].copy(),
            model.body_iquat[body].copy(), model.body_inertia[body].copy())


class RobotInterface: 
    """
    High-level interface for Franka FR3 robot control.
    
    This class provides a unified interface for both real robot control (via pylibfranka)
    and simulation (via MuJoCo). It handles low-level communication, state synchronization,
    and kinematics/dynamics computation.
    
    Attributes:
        real (bool): True if connected to real robot, False for simulation
        model (mujoco.MjModel): MuJoCo model for kinematics/dynamics
        data (mujoco.MjData): MuJoCo data structure with current state
        robot (pylibfranka.Robot): Real robot interface (if real=True)
        torque_controller: Active torque control interface
        site_name (str): Name of end-effector site in MuJoCo model
        site_id (int): MuJoCo site ID for end-effector
        viewer (mujoco.viewer): MuJoCo viewer window (if real=False)
        
    Examples:
        Real robot:
            >>> robot = RobotInterface("172.16.0.2")
            >>> robot.start()
            >>> state = robot.state
            >>> robot.step(np.zeros(7))  # Send zero torques
            >>> robot.stop()
            
        Simulation:
            >>> robot = RobotInterface(None)
            >>> state = robot.state
            >>> robot.step(np.zeros(7))  # Updates MuJoCo simulation
        
    Caveats:
        - Must call start() before step() on real robot
        - Collision behavior is set to high thresholds by default
        - State is synced from robot on every access (thread-safe)
        - MuJoCo model must match real robot configuration
    """

    def __init__(self, ip = None, read_tool = True):
        """
        Initialize robot interface.
        
        Args:
            ip (str | None): Robot IP address (e.g., "172.16.0.2") for real robot,
                           or None for simulation mode.
            read_tool (bool): Read the end-effector profile active in Desk (robot.tool),
                           which FrankaController.activate() checks configurations
                           against, with the Desk credentials saved by aiofranka; it
                           waits up to 3 s for Desk and never takes the control token
                           
        Raises:
            ModuleNotFoundError: If pylibfranka is not installed (real robot mode)
            ConnectionError: If cannot connect to robot at given IP
            
        Note:
            In real mode, collision thresholds are set to [100.0] * 7 for joints
            and [100.0] * 6 for Cartesian space. Adjust via robot.robot.set_collision_behavior()
            for more conservative behavior.
        """

        self.real = ip is not None
        self.ip = ip

        self.model = mujoco.MjModel.from_xml_path(f"{CUR_DIR}/model/fr3.xml")
        self.data = mujoco.MjData(self.model)

        self.torque_controller = None
        # Last pylibfranka.RobotState read from the robot (None in simulation).
        self.robot_state = None
        # End-effector profile active in Desk when connecting (an aiofranka.Tool), which
        # FrankaController.activate() checks configurations against, or None if Desk
        # could not be read (tool_error says why) or in simulation.
        self.tool = None
        self.tool_error = "MuJoCo has no Desk" if ip is None else "RobotInterface(read_tool=False) did not read Desk"

        # End-effector site we wish to control.
        self.site_name = "attachment_site"
        self.site_id = self.model.site(self.site_name).id

        # Load on the flange set with set_load(); the payload merged into the last
        # link of the MuJoCo model, see sync_payload(); and the inertial of the last
        # link without it.
        self.load = {"mass": 0.0, "com": np.zeros(3), "inertia": np.zeros((3, 3))}
        self.payload = {"mass": 0.0, "com": np.zeros(3), "inertia": np.zeros((3, 3))}
        self._last_link_inertial = link_inertial(self.model)

        # The first calls of these take a few hundred microseconds, which would miss
        # the first control cycles. Make them once before the 1 kHz loop starts.
        mujoco.mj_forward(self.model, self.data)
        _ = self.data.site(self.site_id).xmat
        jac = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)
        mujoco.mj_fullM(self.model, self.data, np.zeros((self.model.nv, self.model.nv)))

        if self.real: 
            pylibfranka = require_pylibfranka()
            self.robot = pylibfranka.Robot(ip, pylibfranka.RealtimeConfig.kIgnore)

            self.robot.set_collision_behavior(
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            )
            # A load from an earlier connection would add to the active end-effector
            # profile in Desk, e.g. one activated with aiofranka.load_tool().
            self.robot.set_load(0.0, [0.0, 0.0, 0.0], [0.0] * 9)

            self.sync_mj()
            self.sync_payload()

            # Read Desk now: during control, an HTTP request would stall the 1 kHz loop.
            if read_tool:
                try:
                    from aiofranka.tools import active_tool
                    self.tool = active_tool(ip, timeout=3.0)
                except Exception as error:
                    self.tool_error = f"could not read Desk's active end-effector profile: {error}"

        else: 
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.data.qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync() 
        
    def start(self): 
        """
        Start torque control mode on the real robot.
        
        This must be called before sending torque commands. Does nothing in simulation.
        
        Raises:
            RuntimeError: If robot is not ready or already in control mode
            
        Caveat:
            After calling start(), you must send torque commands at ~1kHz to maintain
            control. Use FrankaController for automatic control loop management.
        """
        if self.real:
            self.torque_controller = self.robot.start_torque_control()

    def stop(self): 
        """
        Stop torque control mode on the real robot.
        
        This gracefully terminates the control session. Does nothing in simulation.
        
        Caveat:
            Robot will hold position briefly then release brakes. Ensure robot
            is in a safe configuration before stopping.
        """
        native = self._native_loop()
        if native is not None:
            # The loop must not read or write the motion this stops.
            native._loop.request_stop()
            native._loop.join()
        if self.real:
            self.robot.stop()
            # Release the torque controller right away. Its destructor cleans up the
            # stopped motion, which should not happen at an arbitrary later point,
            # e.g. when the interpreter exits.
            self.torque_controller = None

    def set_load(self, mass, com=(0.0, 0.0, 0.0), inertia=(0.0, 0.0, 0.0)):
        """
        Set the load attached to the flange, e.g. a tool without a gripper.

        The load is set on the robot, whose internal controller compensates its
        gravity, and merged into the last link of the MuJoCo model with
        sync_payload(), so the mass matrix used by the OSC and the simulated
        dynamics include it. The flange frame is the attachment_site frame. A mass
        of 0 removes the load.

        Args:
            mass (float): Load mass [kg]
            com (array-like): Center of mass in the flange frame [m] (3,)
            inertia (array-like): Inertia about the center of mass in the flange
                frame [kg m^2], as a (3, 3) tensor or its diagonal (3,)

        Raises:
            RuntimeError: If called while torque control is running

        Caveats:
            - Call before start(); the robot rejects the load during control
            - libfranka adds the load to the active end-effector profile in Desk
            - The load lasts until the next connection, which resets it. To keep a
              tool, save it as a profile with aiofranka.save_tool() and activate it
              with aiofranka.load_tool().
            - The robot rejects a mass without inertia
        """
        com = np.asarray(com, dtype=float)
        inertia = np.asarray(inertia, dtype=float)
        if inertia.shape == (3,):
            inertia = np.diag(inertia)

        if self.real:
            if self.torque_controller is not None:
                raise RuntimeError("set_load() must be called before start()")
            self.robot.set_load(float(mass), com.tolist(), inertia.flatten(order="F").tolist())

        self.load = {"mass": float(mass), "com": com, "inertia": inertia}
        if self.real:
            # The robot state shows the new load only after a few cycles.
            deadline = time.time() + 1.0
            self.sync_mj()
            while abs(self.robot_state.m_load - mass) > 1e-6 and time.time() < deadline:
                self.sync_mj()
        self.sync_payload()

    def sync_payload(self):
        """
        Merge the payload that the robot compensates into the MuJoCo model.

        On the real robot, the payload is the end effector configured in Desk plus
        the load set with set_load(), from the last robot state read. In
        simulation, it is the load. It replaces the payload merged before, so the
        mass matrix used by the OSC matches what the robot compensates.

        RobotInterface calls this when it connects and in set_load(). Call it after
        changing the end effector in Desk while connected, after sync_mj().
        """
        if self.real:
            state = self.robot_state
            mass = float(state.m_total)
            com = np.array(state.F_x_Ctotal)
            inertia = np.array(state.I_total).reshape(3, 3, order="F")
        else:
            mass, com, inertia = self.load["mass"], self.load["com"], self.load["inertia"]

        merge_payload(self.model, self._last_link_inertial, mass, com, inertia, self.site_id)

        # mj_setConst resets the state to qpos0, so keep the current one.
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos, self.data.qvel = qpos, qvel
        mujoco.mj_forward(self.model, self.data)

        self.payload = {"mass": mass, "com": com, "inertia": inertia}

        # A NativeFrankaController computes with its own copy of the model.
        native = self._native()
        if native is not None:
            native._model_changed()

    def _native(self):
        """The NativeFrankaController that runs, or ran, on this robot, or None."""
        ref = getattr(self, "_native_ref", None)
        return ref() if ref is not None else None

    def _native_loop(self):
        """The NativeFrankaController whose loop runs now and reads every robot state, or None."""
        native = self._native()
        return native if native is not None and native._owns_connection() else None

    def sync_mj(self):
        """ Sync mujoco state with real robot state """

        native = self._native_loop()
        if native is not None:
            # The native loop reads every state; take its last one instead.
            native._sync_world()
            return

        if self.torque_controller is None:
            robot_state = self.robot.read_once()
        else:
            robot_state, _ = self.torque_controller.readOnce()
        self.robot_state = robot_state

        self.data.qpos = np.array(robot_state.q)
        self.data.qvel = np.array(robot_state.dq)
        self.data.ctrl = np.array(robot_state.tau_J_d)
        mujoco.mj_forward(self.model, self.data)

    @property
    def state(self):
        """
        Get current robot state with kinematics and dynamics.

        Returns:
            dict: Dictionary containing:
                - qpos (np.ndarray): Joint positions [rad] (7,)
                - qvel (np.ndarray): Joint velocities [rad/s] (7,)
                - ee (np.ndarray): End-effector pose as 4x4 homogeneous transform
                                  [[R, p], [0, 1]] where R is rotation, p is position
                - jac (np.ndarray): End-effector Jacobian (6, 7) - [linear; angular]
                - mm (np.ndarray): Joint-space mass matrix (7, 7)
                - last_torque (np.ndarray): Last commanded torques [Nm] (7,)

        Note:
            State is synchronized from real robot on every access. MuJoCo model
            is updated with latest robot state before computing kinematics/dynamics.

        Example:
            >>> state = robot.state
            >>> print(f"Joint 1 position: {state['qpos'][0]:.3f} rad")
            >>> print(f"EE position: {state['ee'][:3, 3]}")
            >>> print(f"EE orientation: {state['ee'][:3, :3]}")
        """

        if self.real:
            self.sync_mj()

        state = {
            "qpos": np.array(self.data.qpos),
            "qvel": np.array(self.data.qvel),
            "ee": self._ee(),
            "jac": self._jacobian(),
            "mm": self._mass_matrix(),
            "last_torque": np.array(self.data.ctrl),
        }

        return state

    @property
    def state_minimal(self):
        """
        Get minimal robot state (fast, no kinematics/dynamics computation).

        Use this for impedance/PID control where Jacobian and mass matrix
        are not needed. ~10x faster than full state property.

        Returns:
            dict: Dictionary containing only:
                - qpos (np.ndarray): Joint positions [rad] (7,)
                - qvel (np.ndarray): Joint velocities [rad/s] (7,)
                - last_torque (np.ndarray): Last commanded torques [Nm] (7,)
        """
        native = self._native_loop()
        if native is not None:
            # The native loop reads every state; take its last one instead.
            state = native.state
            if state is not None:
                return {key: state[key] for key in ("qpos", "qvel", "last_torque")}
        if self.real:
            if self.torque_controller is None:
                robot_state = self.robot.read_once()
            else:
                robot_state, _ = self.torque_controller.readOnce()

            return {
                "qpos": np.array(robot_state.q),
                "qvel": np.array(robot_state.dq),
                "last_torque": np.array(robot_state.tau_J_d),
            }
        else:
            # Simulation: still need mj_forward for accurate velocities
            mujoco.mj_forward(self.model, self.data)
            return {
                "qpos": np.array(self.data.qpos),
                "qvel": np.array(self.data.qvel),
                "last_torque": np.array(self.data.ctrl),
            } 

    def _mass_matrix(self): 
        """ Compute mass matrix at current state """

        mm = np.zeros((7,7))
        mujoco.mj_fullM(self.model, self.data, mm)
        return mm

    def _ee(self):
        ee_xyz = self.data.site(self.site_id).xpos
        ee_mat = self.data.site(self.site_id).xmat.reshape(3,3)
        ee = np.eye(4)
        ee[:3, :3] = ee_mat
        ee[:3, 3] = ee_xyz

        return ee

        
    def _jacobian(self):
        jac = np.zeros((6, 7))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)
        return jac

    def step(self, torque: np.ndarray): 
        """
        Send torque command to robot or step simulation.
        
        Args:
            torque (np.ndarray): Joint torques [Nm] (7,)
            
        Raises:
            RuntimeError: If real robot not started or communication error
            
        Note:
            - Real robot: Sends torque command via pylibfranka at current timestep
            - Simulation: Updates MuJoCo with torques and advances one timestep
            
        Caveats:
            - Must be called at ~1kHz for real robot to maintain control
            - Large torque changes may trigger safety limits
            - Torques should respect robot limits: ``|tau_i| < 87 Nm`` for joints 1-4,
              ``|tau_i| < 12 Nm`` for joints 5-7
              
        Example:
            >>> # Send gravity compensation torques
            >>> torque = robot.state['mm'] @ np.array([0, 0, 0, 0, 0, 0, -9.81])
            >>> robot.step(torque)
        """

        if self._native_loop() is not None:
            raise RuntimeError("A NativeFrankaController's loop sends the torques while it runs; "
                               "command them with controller.torque in torque mode")
        if self.real:
            import pylibfranka
            torque_command = pylibfranka.Torques(torque.tolist())
            torque_command.motion_finished = False
            self.torque_controller.writeOnce(torque_command)
        else: 
            self.data.ctrl = torque
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

if __name__ == "__main__": 

    robot = RobotInterface("172.16.0.2")
    while True: 

        zero_torque = np.zeros(7)
        robot.step(zero_torque)
        time.sleep(0.1)
        print(zero_torque)
