"""
Identification of the tool attached to the flange.

The robot compensates the gravity of the arm and of the payload: the end effector
configured in Desk plus the load set with RobotInterface.set_load(). If the
payload does not match the tool, the arm drifts in torque control, e.g. with zero
torques. identify_payload() estimates the mass and center of mass of the tool from
the joint torques at rest in a set of poses, starting from any payload setting,
e.g. none.

At rest, the measured joint torques minus the gravity torques of the robot model,
tau_ext_hat_filtered, are the gravity torques of the part of the load the robot
does not know about, plus stiction. Some joint friction acts past the torque
sensors: at rest, they also see part of the torque with which the controller holds
the joint against stiction, which on an FR3 was a few tenths of a Nm, flipping
sign with the side the joint came from. So identify_payload() approaches each pose
from both sides and averages. The gravity torques are linear in the mass m and the
first moment m * c of the unknown part:

    tau = g * [J_v^T e_z | -J_w^T [e_z]x R] @ [m, m * c]

where J_v and J_w are the linear and angular Jacobian and R the orientation of the
flange in the base frame, and c is the center of mass in the flange frame. With
poses in which the flange points in different directions, least squares gives m
and c. The inertia does not change the gravity torques, so it cannot be identified
at rest. It does not matter for the robot's gravity compensation either.
"""

import asyncio
import contextlib
import io
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).parent.resolve() / "model" / "fr3.xml"
SITE_NAME = "attachment_site"

# Gravitational acceleration the robot uses [m/s^2].
GRAVITY = 9.81

# Default half-widths [rad] of the joint ranges around the current pose that the
# planned poses are sampled from. Joint 1 does not change the gravity torques.
# The wrist joints change the flange orientation, which makes the center of mass
# identifiable.
DEFAULT_SPANS = (0.0, 0.3, 0.3, 0.3, 1.3, 1.0, 2.6)

# Distance [rad] to the joint limits that planned poses keep.
JOINT_LIMIT_MARGIN = 0.1

# Offsets [rad] from which identify_payload() approaches each pose, from both sides,
# so that stiction, which flips sign with the side, averages out. Joint 1 carries
# no gravity torque.
APPROACH = np.array([0.0, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

# Skew-symmetric matrix of the z-axis of the base frame, [e_z]x v = e_z x v.
_EZ_CROSS = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@lru_cache(maxsize=None)
def _kinematics():
    # Loading the model takes about 0.2 s.
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.MjData(model), model.site(SITE_NAME).id


def _robot_kinematics(robot):
    """Kinematics of a RobotInterface's model, whose loading is done already."""
    return robot.model, mujoco.MjData(robot.model), robot.site_id


def payload_regressor(qpos, kinematics=None):
    """
    Regressor of the gravity torques of a load on the flange.

    Args:
        qpos (array-like): Joint positions [rad] (7,)
        kinematics (tuple | None): MuJoCo model, data and flange site id to
            compute it with (default: a model of the FR3)

    Returns:
        np.ndarray: Matrix Y (7, 4) such that Y @ [m, m * cx, m * cy, m * cz] are
            the joint torques [Nm] that hold a load of mass m [kg] with center of
            mass c [m] in the flange frame against gravity.
    """
    model, data, site_id = kinematics or _kinematics()
    data.qpos[:7] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    rot = data.site_xmat[site_id].reshape(3, 3)
    return GRAVITY * np.hstack([jacp[2, :7, None], -jacr[:, :7].T @ _EZ_CROSS @ rot])


def _design(qpos, fit_bias, kinematics=None):
    """Stacked regressors (n, 7, p), with a torque offset per joint if fit_bias."""
    Y = np.stack([payload_regressor(q, kinematics) for q in qpos])
    if fit_bias:
        Y = np.concatenate([Y, np.broadcast_to(np.eye(7), (len(Y), 7, 7))], axis=2)
    return Y


class _Progress:
    """
    Progress bar like FrankaController.move()'s, advanced by the planned time of
    each step, with an ETA scaled by how much longer the steps took so far.
    """

    def __init__(self, total, n_poses, enabled):
        self.total = total
        self.n_poses = n_poses
        self.enabled = enabled
        self.done = 0.0
        self.measured = 0
        self.t0 = time.perf_counter()
        self.show()

    def advance(self, seconds, measured=0):
        self.done += seconds
        self.measured += measured
        self.show()

    def show(self):
        if not self.enabled:
            return
        fraction = min(self.done / self.total, 1.0)
        elapsed = time.perf_counter() - self.t0
        eta = max(self.total - self.done, 0.0)
        if self.done > 0:
            eta *= elapsed / self.done
        filled = int(fraction * 20)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"\r  Identifying [{bar}] {fraction:4.0%}  {self.measured}/{self.n_poses} poses"
              f"  ETA {int(eta) // 60}:{int(eta) % 60:02d} ", end="", flush=True)

    def finish(self):
        if self.enabled:
            print()


@dataclass
class PayloadEstimate:
    """
    Tool on the flange estimated by identify_payload() or fit_payload().

    The estimate is the whole tool: the payload that was set during the
    measurements plus the correction the measurements found.

    Attributes:
        mass (float): Tool mass [kg]
        com (np.ndarray): Center of mass in the flange frame [m] (3,)
        mass_std (float): Standard error of the mass [kg]
        com_std (np.ndarray): Standard error of the center of mass [m] (3,)
        previous (dict): Payload that was set during the measurements, with
            "mass" and "com"
        bias (np.ndarray): Torque offsets per joint [Nm] (7,), e.g. from the
            torque sensors. The payload cannot compensate them.
        rms_before (np.ndarray): RMS of the residual torques at rest with the
            previous payload [Nm] (7,)
        rms_after (np.ndarray): RMS of the residual torques at rest predicted with
            this tool [Nm] (7,). It includes the offsets in bias.
        condition (float): Condition number of the least-squares problem. Large
            values, e.g. > 100, mean the poses do not tell the parameters apart.
        qpos (np.ndarray): Joint positions of the measurements [rad] (n, 7)
        residuals (np.ndarray): Measured residual torques at rest [Nm] (n, 7)
    """

    mass: float
    com: np.ndarray
    mass_std: float
    com_std: np.ndarray
    previous: dict
    bias: np.ndarray
    rms_before: np.ndarray
    rms_after: np.ndarray
    condition: float
    qpos: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)

    def __str__(self):
        com = ", ".join(f"{axis} {1e3 * c:+.1f} ± {1e3 * s:.1f}"
                        for axis, c, s in zip("xyz", self.com, self.com_std))
        # Joint 1 carries no gravity torque.
        before, after = (np.sqrt(np.mean(rms[1:] ** 2)) for rms in (self.rms_before, self.rms_after))
        return "\n".join([
            "Estimated tool, in the flange frame:",
            f"  mass            {1e3 * self.mass:.1f} ± {1e3 * self.mass_std:.1f} g"
            f"   (previous payload {1e3 * self.previous['mass']:.1f} g)",
            f"  center of mass  {com} mm",
            f"  fit             residual torque {before:.2f} -> {after:.2f} Nm RMS over {len(self.qpos)} poses",
        ])


def fit_payload(qpos, residuals, previous=None, fit_bias=True, kinematics=None):
    """
    Estimate the tool on the flange from residual torques at rest.

    Solves for the mass and first moment of the unmodeled part of the tool, and
    with fit_bias a constant torque offset per joint, by least squares. A second
    pass weights the joints by their residual noise, so that model errors of the
    shoulder joints do not swamp the wrist joints, which see the center of mass.

    Args:
        qpos (array-like): Joint positions [rad] (n, 7)
        residuals (array-like): Residual torques at rest [Nm] (n, 7): the
            measured joint torques minus the gravity torques of the arm and the
            previous payload, e.g. tau_ext_hat_filtered
        previous (dict | None): Payload that was set during the measurements,
            with "mass" [kg] and "com" [m] (default: none)
        fit_bias (bool): Estimate a torque offset per joint (default: True)
        kinematics (tuple | None): See payload_regressor()

    Returns:
        PayloadEstimate: Estimated tool, i.e. previous plus the correction

    Raises:
        ValueError: If there are too few poses for the parameters
    """
    qpos = np.asarray(qpos, dtype=float).reshape(-1, 7)
    residuals = np.asarray(residuals, dtype=float).reshape(-1, 7)
    if previous is None:
        previous = {"mass": 0.0, "com": np.zeros(3)}
    previous = {"mass": float(previous["mass"]), "com": np.asarray(previous["com"], dtype=float)}

    n = len(qpos)
    design = _design(qpos, fit_bias, kinematics)
    p = design.shape[2]
    dof = 7 * n - p
    if n < 2 or dof < 1:
        raise ValueError(f"{n} poses are too few to estimate {p} parameters")

    A = design.reshape(-1, p)
    r = residuals.reshape(-1)
    theta = np.linalg.lstsq(A, r, rcond=None)[0]

    # Weight each joint by the noise of its residuals.
    error = (r - A @ theta).reshape(n, 7)
    sigma = np.maximum(np.sqrt((error ** 2).sum(axis=0) * 7 / dof), 1e-3)
    weights = np.tile(1.0 / sigma, n)
    Aw = A * weights[:, None]
    rw = r * weights
    theta = np.linalg.lstsq(Aw, rw, rcond=None)[0]
    error = rw - Aw @ theta
    cov = (error @ error / dof) * np.linalg.pinv(Aw.T @ Aw)
    condition = float(np.linalg.cond(Aw / np.linalg.norm(Aw, axis=0)))

    delta = theta[:4]
    mass = previous["mass"] + delta[0]
    moment = previous["mass"] * previous["com"] + delta[1:]
    if abs(mass) > 1e-3:
        com = moment / mass
        # Propagate the covariance of (m, m * c) to c = (m * c) / m.
        jac = np.hstack([-moment[:, None] / mass ** 2, np.eye(3) / mass])
        com_std = np.sqrt(np.diag(jac @ cov[:4, :4] @ jac.T))
    else:
        com = np.zeros(3)
        com_std = np.full(3, np.inf)

    unmodeled = design[:, :, :4] @ delta
    return PayloadEstimate(
        mass=float(mass),
        com=com,
        mass_std=float(np.sqrt(cov[0, 0])),
        com_std=com_std,
        previous=previous,
        bias=theta[4:] if fit_bias else np.zeros(7),
        rms_before=np.sqrt(np.mean(residuals ** 2, axis=0)),
        rms_after=np.sqrt(np.mean((residuals - unmodeled) ** 2, axis=0)),
        condition=condition,
        qpos=qpos,
        residuals=residuals,
    )


@lru_cache(maxsize=8)
def _collision_model(tool_length, tool_radius, floor, clearance):
    """
    The arm with a cylinder around the tool and a floor, for collision checks.

    Contacts are reported for geoms closer than clearance to the tool or floor.
    """
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    site = next(s for s in spec.sites if s.name == SITE_NAME)
    tip = site.pos + np.array([0.0, 0.0, tool_length])
    spec.body("fr3_link7").add_geom(
        name="tool",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=np.concatenate([site.pos, tip]),
        size=[tool_radius, 0.0, 0.0],
        margin=clearance,
        mass=0.0,
    )
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0.0, 0.0, floor],
        size=[0.0, 0.0, 1.0], margin=clearance,
    )
    spec.add_exclude(bodyname1="world", bodyname2="fr3_link0")
    model = spec.compile()
    return model, mujoco.MjData(model)


def _is_clear(model, data, qpos):
    data.qpos[:7] = qpos
    mujoco.mj_fwdPosition(model, data)
    # MuJoCo adds the margins of two geoms, so it reports the tool from as far as twice
    # the clearance from the floor: only contacts within the larger margin count.
    margin = model.geom_margin
    return all(c.dist >= max(margin[c.geom1], margin[c.geom2]) for c in data.contact[:data.ncon])


def _closest(model, data, qpos, clearance):
    """Describes the closest pair of parts at qpos that are too close to each other."""
    _is_clear(model, data, qpos)

    def part(geom):
        name = model.geom(geom).name
        if name in ("tool", "floor"):
            return f"the {name}"
        return "link " + model.body(model.geom_bodyid[geom]).name.removeprefix("fr3_link")

    contact = min(data.contact[:data.ncon], key=lambda c: c.dist)
    a, b = part(contact.geom1), part(contact.geom2)
    if contact.dist <= 0:
        return f"{a} and {b} collide"
    return f"{a} is {100 * contact.dist:.0f} cm from {b}, closer than the {100 * clearance:.0f} cm clearance"


def _is_path_clear(model, data, start, end, step=0.05):
    n = max(int(np.ceil(np.max(np.abs(end - start)) / step)), 1)
    return all(_is_clear(model, data, start + s * (end - start)) for s in np.linspace(0, 1, n + 1))


@dataclass
class _Plan:
    """
    Motion of identify_payload().

    Attributes:
        center (np.ndarray): Joint positions the motion starts and ends at [rad] (7,)
        waypoints (np.ndarray): Joint positions to move to in order, along straight
            lines [rad] (m, 7)
        pose (np.ndarray): For each waypoint, the index of the pose measured there,
            or -1 to pass through it (m,). identify_payload() averages the
            measurements of each pose.
    """

    center: np.ndarray
    waypoints: np.ndarray
    pose: np.ndarray

    @property
    def poses(self):
        """The measured poses [rad] (n, 7)."""
        return np.array([self.waypoints[np.argmax(self.pose == i)] for i in range(self.pose.max() + 1)])


def _approaches(pose, index):
    """Waypoints that measure pose after approaching it from both sides."""
    return [pose + APPROACH, pose, pose - APPROACH, pose], [-1, index, -1, index]


def _plan(
    center,
    n_poses=16,
    tool_length=0.2,
    tool_radius=0.1,
    floor=0.0,
    clearance=0.05,
    fit_bias=True,
    spans=DEFAULT_SPANS,
    n_candidates=2000,
    seed=0,
):
    """
    Plan the motion of identify_payload() around center.

    Samples poses around center within the joint limits and keeps those in which
    the arm and a cylinder around the tool neither collide with the arm nor come
    closer than clearance to each other or to the floor, also at
    the approach offsets APPROACH on both sides of the pose. From those, it greedily
    picks the poses that make the parameters best identifiable (D-optimal design)
    and orders them to keep the motions short. All straight paths between the
    waypoints are checked; where one is not clear, the motion goes through center.

    It computes for a few tenths of a second, which would stall a running 1 kHz
    control loop.

    Returns:
        _Plan: The motion

    Raises:
        ValueError: If center is not clear, or too few sampled poses are
    """
    center = np.asarray(center, dtype=float)
    spans = np.asarray(spans, dtype=float)
    model, data = _collision_model(
        float(tool_length), float(tool_radius), float(floor), float(clearance)
    )
    if not _is_clear(model, data, center):
        raise ValueError(
            f"At the current pose, {_closest(model, data, center, clearance)}. Move the arm "
            "to an open pose, or reduce tool_length, tool_radius or clearance."
        )

    # Poses whose approach offsets stay within the joint limits too.
    limits = model.jnt_range[:7]
    low = np.maximum(center - spans, limits[:, 0] + JOINT_LIMIT_MARGIN + APPROACH)
    high = np.minimum(center + spans, limits[:, 1] - JOINT_LIMIT_MARGIN - APPROACH)
    low, high = np.minimum(low, center), np.maximum(high, center)
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low, high, size=(n_candidates, 7))
    candidates = np.array([
        q for q in samples
        if all(_is_clear(model, data, w) for w in (q, q + APPROACH, q - APPROACH))
    ])
    if len(candidates) < n_poses:
        raise ValueError(
            f"Only {len(candidates)} of {n_candidates} sampled poses are clear. "
            "Increase spans or n_candidates, or reduce the tool size or clearance."
        )

    def reachable(q):
        """Whether the paths from center to q and its approach are clear."""
        return (_is_path_clear(model, data, center, q)
                and _is_path_clear(model, data, center, q + APPROACH)
                and _is_path_clear(model, data, q + APPROACH, q - APPROACH))

    # Greedy D-optimal design: add the pose that increases log det of the
    # information matrix most. Its paths are checked lazily.
    design = _design(candidates, fit_bias)
    design = design / np.sqrt(np.mean(design ** 2, axis=(0, 1)))
    p = design.shape[2]
    info = 1e-6 * np.eye(p)
    available = np.ones(len(candidates), dtype=bool)
    chosen = []
    while len(chosen) < n_poses and available.any():
        gain = np.linalg.slogdet(np.eye(7) + design @ np.linalg.solve(info, design.transpose(0, 2, 1)))[1]
        gain[~available] = -np.inf
        best = int(np.argmax(gain))
        available[best] = False
        if reachable(candidates[best]):
            chosen.append(best)
            info += design[best].T @ design[best]
    if len(chosen) < n_poses:
        raise ValueError(f"Only {len(chosen)} poses have clear paths from the center pose")

    # Visit the poses nearest neighbor first, going through center where the
    # straight path to the next approach is not clear.
    remaining = list(chosen)
    waypoints, index = [], []
    current = center
    while remaining:
        nearest = min(remaining, key=lambda i: np.max(np.abs(candidates[i] - current)))
        remaining.remove(nearest)
        target = candidates[nearest]
        if not _is_path_clear(model, data, current, target + APPROACH):
            waypoints.append(center.copy())
            index.append(-1)
        more_waypoints, more_index = _approaches(target, len(chosen) - len(remaining) - 1)
        waypoints += more_waypoints
        index += more_index
        current = target
    return _Plan(center, np.array(waypoints), np.array(index))


def _move_duration(start, end, speed):
    # Peak velocity of a quintic time scaling is 1.875 times the average.
    return max(1.875 * np.max(np.abs(end - start)) / speed, 0.5)


async def _move_to(controller, target, speed):
    """Move in a straight line in joint space with a quintic time scaling."""
    with controller.state_lock:
        start = np.array(controller.q_desired, dtype=float)
    target = np.asarray(target, dtype=float)
    duration = _move_duration(start, target, speed)
    t0 = time.perf_counter()
    while True:
        s = min((time.perf_counter() - t0) / duration, 1.0)
        with controller.state_lock:
            controller.q_desired = start + s ** 3 * (10 - 15 * s + 6 * s ** 2) * (target - start)
        if s >= 1.0:
            return
        await asyncio.sleep(0.01)


async def _settle(controller, wait, tolerance=0.01, hold=0.2, timeout=5.0):
    """Wait for wait seconds, then until all joint speeds stay below tolerance."""
    await asyncio.sleep(wait)
    deadline = time.perf_counter() + timeout
    still_since = None
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if np.max(np.abs(controller.state["qvel"])) < tolerance:
            still_since = still_since or now
            if now - still_since >= hold:
                return True
        else:
            still_since = None
        await asyncio.sleep(0.005)
    return False


def _read_residual(robot):
    """Joint positions and torques at rest that the robot does not compensate."""
    if robot.real:
        state = robot.robot_state
        # tau_J minus the modeled gravity; it does not subtract the commanded torque.
        return np.array(state.q), np.array(state.tau_ext_hat_filtered)
    # In simulation, the gravity of bodies without gravcomp.
    data = robot.data
    return data.qpos[:7].copy(), data.qfrc_bias[:7] - data.qfrc_gravcomp[:7]


async def _measure(controller, duration, period=0.005):
    qpos, residuals, speed = [], [], 0.0
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        q, r = _read_residual(controller.robot)
        qpos.append(q)
        residuals.append(r)
        speed = max(speed, np.max(np.abs(controller.state["qvel"])))
        await asyncio.sleep(period)
    return np.mean(qpos, axis=0), np.mean(residuals, axis=0), speed


async def _quietly(coroutine):
    """Await coroutine without its prints, e.g. FrankaController.stop()'s."""
    with contextlib.redirect_stdout(io.StringIO()):
        return await coroutine


async def identify_payload(
    controller,
    tool_length=0.2,
    tool_radius=0.1,
    floor=0.0,
    clearance=0.05,
    n_poses=16,
    speed=0.5,
    settle=1.5,
    duration=1.0,
    fit_bias=True,
    verbose=True,
):
    """
    Identify the mass and center of mass of the tool on the flange.

    Plans n_poses poses around the current joint positions, in which the flange
    points in different directions, and moves through them in joint impedance
    control, approaching each from both sides. It measures the residual joint
    torques at rest in each and fits the tool with fit_payload(). It only measures:
    save the result as an end-effector profile in Desk with aiofranka.save_tool()
    and activate it with aiofranka.load_tool(). It starts from the payload the robot
    compensates, e.g. none, so running it again with the new profile active checks
    the result: the correction should be close to zero.

    The poses and the paths between them are checked for collisions of the arm, a
    cylinder around the tool and the floor. Planning computes for a few tenths of
    a second, which would stall a running 1 kHz control loop and make the robot
    abort, so it plans while no control loop runs: a running controller is stopped
    for it, while the robot holds still, and started again.

    Args:
        controller (FrankaController): Controller, started or not. One that was
            not started is started for the identification and stopped afterwards.
        tool_length (float): Length of the tool from the flange along its z-axis
            [m] (default: 0.2)
        tool_radius (float): Radius of the tool around the flange z-axis [m]
            (default: 0.1)
        floor (float): Height of the floor or table in the base frame [m]
            (default: 0.0, the mounting plane of the robot)
        clearance (float): Minimum distance of the tool from the arm, and of the
            arm and tool from the floor [m] (default: 0.05)
        n_poses (int): Number of poses (default: 16)
        speed (float): Peak joint speed of the motions [rad/s] (default: 0.5)
        settle (float): Time to wait in each pose before measuring [s] (default:
            1.5). tau_ext_hat_filtered lags by about 0.3 s.
        duration (float): Time to average the torques over in each pose [s]
            (default: 1)
        fit_bias (bool): Estimate a torque offset per joint (default: True)
        verbose (bool): Show a progress bar and the result (default: True)

    Returns:
        PayloadEstimate: Estimated tool

    Raises:
        ValueError: If too few poses around the current joint positions are clear;
            move the arm to an open pose

    Caveats:
        - Takes about 3 minutes. Start in an open pose, keep other obstacles out of
          reach, and stay close to the e-stop.
        - Afterwards, a running controller is in impedance mode at the start pose.
    """
    robot = controller.robot
    was_running = controller.running
    if was_running:
        await _quietly(controller.stop())
    try:
        plan = _plan(robot.state["qpos"], n_poses=n_poses, tool_length=tool_length,
                     tool_radius=tool_radius, floor=floor, clearance=clearance, fit_bias=fit_bias)
    except Exception:
        if was_running:
            await controller.start()
        raise
    controller.switch("impedance")
    await controller.start()
    try:
        return await _run(controller, plan, speed, settle, duration, fit_bias, verbose)
    finally:
        if not was_running:
            await _quietly(controller.stop())


async def _run(controller, plan, speed, settle, duration, fit_bias, verbose):
    """Move through the plan, measure, and fit, with the controller running."""
    log = print if verbose else (lambda *args, **kwargs: None)
    robot = controller.robot
    previous = dict(robot.payload)
    with controller.state_lock:
        start = np.array(controller.q_desired, dtype=float)

    n_poses = plan.pose.max() + 1
    path = np.vstack([start, plan.center, plan.waypoints, plan.center])
    moves = [_move_duration(a, b, speed) for a, b in zip(path[:-1], path[1:])]
    measuring = settle + duration + 0.3
    progress = _Progress(sum(moves) + np.sum(plan.pose >= 0) * measuring, n_poses, verbose)

    measured = {}
    try:
        await _move_to(controller, plan.center, speed)
        progress.advance(moves[0])
        for waypoint, i, move in zip(plan.waypoints, plan.pose, moves[1:]):
            await _move_to(controller, waypoint, speed)
            progress.advance(move)
            if i < 0:
                continue
            settled = await _settle(controller, settle)
            q, r, peak = await _measure(controller, duration)
            measured.setdefault(i, []).append((q, r) if settled and peak <= 0.02 else None)
            progress.advance(measuring, measured=len(measured[i]) == 2)
    finally:
        await _move_to(controller, plan.center, speed)
        progress.advance(moves[-1])
        progress.finish()

    pairs = [m for m in measured.values() if len(m) == 2 and None not in m]
    if len(pairs) < n_poses:
        log(f"  Skipped {n_poses - len(pairs)} of {n_poses} poses, where the arm did not come to rest")
    qpos = [np.mean([q for q, _ in m], axis=0) for m in pairs]
    residuals = [np.mean([r for _, r in m], axis=0) for m in pairs]
    # The robot's model is loaded already; loading another would stall the loop.
    estimate = fit_payload(qpos, residuals, previous=previous, fit_bias=fit_bias,
                           kinematics=_robot_kinematics(robot))
    log()
    log("\n".join(f"  {line}" for line in str(estimate).splitlines()))
    return estimate
