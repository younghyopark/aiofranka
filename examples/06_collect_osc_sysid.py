#!/usr/bin/env python3
"""
Collect operational space control data to identify the FR3 with 07_fit_osc_sysid.py.

With an OSC configuration (ee_kp, ee_kd, null_kp, null_kd, the null-space target, the TCP,
the policy rate as frequency, and the tool; see aiofranka.config), it plays 13 s blocks of
TCP pose targets around three base poses, two at each, and records every 1 kHz control
cycle to examples/sysid_data/osc_sysid_<date>.npz.

    python examples/06_collect_osc_sysid.py 173.16.0.2 --activate configs/osc.yaml
    mjpython examples/06_collect_osc_sysid.py --activate configs/osc.yaml   # MuJoCo

Each block holds, steps all six axes twice, and plays a multisine and ramps (see
sysid.py) as offsets of the TCP: positions in the base frame, and rotation vectors
in the base frame about the TCP. Steps go up to 30 mm and 0.1 rad; the multisine and ramps
up to 80 mm in x and y, 50 mm in z and 0.25 rad, the multisine at up to 0.3 m/s and 1 rad/s.
All are smaller where an ideal OSC would command more than half the torque limits (with the
task-space inertia of aiofranka's model and the payload) or move a joint faster than half
its speed limit, and the multisine and ramps smaller still until every target, solved with
inverse kinematics, keeps 0.1 rad from the joint limits and the clearance between the arm,
a cylinder around the tool and the floor. The moves between poses are checked too; --plan
only checks.

Without a null_target in the configuration, the base poses are home, left and right, and
each block's null-space target is the posture it starts in. With one (e.g. a policy's),
they are on its branch, with the TCP where the null target puts it, 15 cm toward 0.4 m
height and 12 cm to either side, where the null space is at rest, so activating swings
nothing.

While playing, only the robot's own limits apply: past them its reflexes stop it, and the
rows so far are saved. Ctrl+C holds the arm where it is. Keep a hand on the enabling device.
"""

from __future__ import annotations

import asyncio

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from aiofranka import NativeFrankaController, RobotInterface
from aiofranka.config import add_recording
from aiofranka.payload import MODEL_PATH, _collision_model
from sysid import (CONTROL_HZ, FIELDS, SETTLE, TORQUE_LIMIT, Stop, arguments, collect, held,
                   joint_problem, metadata, move_problems, move_to, multisine, output_path, plan, play,
                   ramps, report, rows, save, summarize)

POSES = {
    "home": [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854],
    "left": [0.6, -0.3, 0.2, -2.2, 0.0, 1.9, -0.2],
    "right": [-0.6, 0.2, -0.2, -1.6, 0.0, 1.8, -1.4],
}
# With a null_target: the TCP's sideways offsets from where the null target puts it [m],
# and NULL_RISE toward NULL_HEIGHT, up from a pose near a table or down from a high one.
NULL_POSES = {"center": [0.0, 0.0], "left": [0.0, 0.12], "right": [0.0, -0.12]}
NULL_RISE, NULL_HEIGHT = 0.15, 0.4  # m
NULL_REST_TOLERANCE = 0.05  # rad the null space may still pull toward the null target at a base pose

STEP_SIZE = np.array([0.03] * 3 + [0.1] * 3)  # largest step [m, rad]
WAVE_SIZE = np.array([0.08, 0.08, 0.05] + [0.25] * 3)  # largest multisine and ramp offset [m, rad]
WAVE_SPEED = np.array([0.3, 0.3, 0.2] + [1.0] * 3)  # largest multisine speed [m/s, rad/s]
SPEED_LIMIT = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])  # FR3 joint speed limits [rad/s]

# What every cycle records, by its name in the file.
RECORDED = FIELDS | {"ee_des": "ee_desired", "ee": "tcp"}


# ── Kinematics ────────────────────────────────────────────────────────────────

def apply(base, offsets):
    """TCP poses (n, 4, 4) at offsets (n, 6) from base."""
    poses = np.tile(np.eye(4), (len(offsets), 1, 1))
    poses[:, :3, 3] = base[:3, 3] + offsets[:, :3]
    poses[:, :3, :3] = Rotation.from_rotvec(offsets[:, 3:]).as_matrix() @ base[:3, :3]
    return poses


def kinematics(model, data, q, tcp=None):
    """The TCP's pose and Jacobian (6, 7, linear rows first) at q; the flange's without a TCP."""
    data.qpos[:7] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    site = model.site("attachment_site").id
    flange = np.eye(4)
    flange[:3, :3], flange[:3, 3] = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
    jac = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site)
    jac = jac[:, :7]
    if tcp is None:
        return flange, jac
    jac[:3] += np.cross(jac[3:].T, flange[:3, :3] @ tcp[:3, 3]).T
    return flange @ tcp, jac


def pose_error(target, pose):
    """Position error and rotation vector from pose to target (6,), for angles below pi."""
    r = target[:3, :3] @ pose[:3, :3].T
    vee = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    sin, cos = np.linalg.norm(vee), 0.5 * (np.trace(r) - 1.0)
    rotation = vee * (np.arctan2(sin, cos) / sin) if sin > 1e-12 else vee
    return np.concatenate([target[:3, 3] - pose[:3, 3], rotation])


def ik(model, data, target, q, q_null, iterations=10):
    """
    Damped least squares inverse kinematics of the flange, pulled toward q_null in the
    null space as the OSC is. Returns the joint positions and the position and rotation errors.
    """
    for _ in range(iterations):
        pose, jac = kinematics(model, data, q)
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-4 * np.eye(6))
        q = q + pinv @ pose_error(target, pose) + 0.05 * (np.eye(7) - pinv @ jac) @ (q_null - q)
    error = pose_error(target, kinematics(model, data, q)[0])
    return q, np.linalg.norm(error[:3]), np.linalg.norm(error[3:])


def null_rest(model, data, q, tcp, q_null):
    """How far the TCP Jacobian's null space would still move the joints toward q_null [rad]."""
    jac = kinematics(model, data, q, tcp)[1]
    return np.abs((np.eye(7) - np.linalg.pinv(jac) @ jac) @ (q_null - q)).max()


# ── Planning ──────────────────────────────────────────────────────────────────

def base_poses(names, tcp, q_null, model, data):
    """The base joint poses, and what is wrong with them."""
    if q_null is None:
        return {name: np.array(POSES[name]) for name in names}, []
    start = kinematics(model, data, q_null, tcp)[0]
    rise = NULL_RISE if start[2, 3] < NULL_HEIGHT else -NULL_RISE
    poses, problems = {}, []
    for name in names:
        target = start.copy()
        target[:3, 3] += [*NULL_POSES[name], rise]
        poses[name], position_error, rotation_error = ik(model, data, target @ np.linalg.inv(tcp), q_null, q_null,
                                                  iterations=300)
        if position_error > 1e-4 or rotation_error > 1e-3:
            problems.append(f"{name}: the TCP {np.round(target[:3, 3], 3).tolist()} is out of reach")
    return poses, problems


def waves():
    """The multisine and the ramps at full size."""
    return multisine(WAVE_SIZE, WAVE_SPEED), ramps(WAVE_SIZE)


def ideal_response(offsets, ee_kp, ee_kd):
    """Acceleration commands and velocities of an ideal OSC, each axis a unit mass, following offsets at 1 kHz."""
    x, v = np.zeros(6), np.zeros(6)
    acceleration, velocity = np.empty_like(offsets), np.empty_like(offsets)
    for k, target in enumerate(offsets):
        acceleration[k] = a = ee_kp * (target - x) - ee_kd * v
        velocity[k] = v
        v = v + a / CONTROL_HZ
        x = x + v / CONTROL_HZ
    return acceleration, velocity


def sizes(model, poses, tcp, ee_kp, ee_kd, rate):
    """
    By pose, the steps' size and the scale (at most 1) of the multisine and ramps, with the
    joint torques and speeds they reach. The OSC commands the torques J' Λ (ee_kp e - ee_kd v),
    J the TCP's Jacobian and Λ the task-space inertia: a step of all axes at once, signs
    adding up, must stay within half the torque limits, and an ideal OSC following the waves
    within half the torque limits and half the joint speed limits (J⁺ v).
    """
    data = mujoco.MjData(model)
    acceleration, velocity = ideal_response(held(np.concatenate(waves()), rate), ee_kp, ee_kd)
    out = {}
    for name, q in poses.items():
        jac = kinematics(model, data, q, tcp)[1]
        mujoco.mj_forward(model, data)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        torque_map = jac.T @ np.linalg.inv(jac @ np.linalg.inv(mass[:7, :7]) @ jac.T)  # J' Λ (7, 6)
        per_step = np.abs(torque_map * ee_kp)  # torques per unit step of each axis
        step = STEP_SIZE * min(1.0, (0.5 * TORQUE_LIMIT / (per_step @ STEP_SIZE)).min())
        torque = np.abs(acceleration @ torque_map.T).max(0)
        speed = np.abs(velocity @ np.linalg.pinv(jac).T).max(0)
        scale = min(1.0, (0.5 * TORQUE_LIMIT / torque).min(), (0.5 * SPEED_LIMIT / speed).min())
        out[name] = {"step": step, "step_torque": per_step @ step, "scale": scale,
                     "wave_torque": scale * torque, "wave_speed": scale * speed, "shrunk": 1.0}
    return out


def target_problem(model, data, q0, offsets, tcp, q_null, clearance):
    """
    What is wrong with the TCP targets at offsets (n, 6) from base pose q0, solved in order
    with inverse kinematics pulled toward the null-space target (q0 without one), or None.
    """
    to_flange = np.linalg.inv(tcp)
    base = kinematics(model, data, q0, tcp)[0]
    toward = q0 if q_null is None else q_null
    q, _, _ = ik(model, data, base @ to_flange, q0, toward, iterations=300)  # where the null space settles
    changes = np.r_[True, np.any(offsets[1:] != offsets[:-1], axis=1)]
    for offset in offsets[changes]:
        q, position_error, rotation_error = ik(model, data, apply(base, offset[None])[0] @ to_flange, q, toward)
        if position_error > 1e-3 or rotation_error > 5e-3:
            return f"a target is out of reach (offset {np.round(offset, 3).tolist()})"
        problem = joint_problem(model, data, q, clearance)
        if problem:
            return problem
    return None


def shrink_waves(limits, poses, rate, tcp, q_null, model, data, clearance):
    """Scale each pose's multisine and ramps down, by bisection, until all their targets pass target_problem()."""
    full = np.concatenate(waves())[::CONTROL_HZ // rate]
    for name, q0 in poses.items():
        low, high = 0.0, limits[name]["scale"]
        if target_problem(model, data, q0, full * high, tcp, q_null, clearance) is None:
            continue
        for _ in range(6):
            middle = 0.5 * (low + high)
            if target_problem(model, data, q0, full * middle, tcp, q_null, clearance) is None:
                low = middle
            else:
                high = middle
        shrunk = low / limits[name]["scale"]
        limits[name].update(scale=low, shrunk=shrunk, wave_torque=limits[name]["wave_torque"] * shrunk,
                     wave_speed=limits[name]["wave_speed"] * shrunk)


def problems(blocks, poses, tcp, q_null, model, data, clearance, start):
    """What is wrong with the base poses, the targets and the moves from start (the arm's pose, if known)."""
    out = []
    for pose, q0 in poses.items():
        if q_null is not None and (rest := null_rest(model, data, q0, tcp, q_null)) > NULL_REST_TOLERANCE:
            out.append(f"{pose}: the null space is not at rest there ({rest:.2f} rad toward the null target)")
        offsets = np.concatenate([b.offsets for b in blocks if b.pose == pose])
        problem = target_problem(model, data, q0, offsets, tcp, q_null, clearance)
        if problem:
            out.append(f"{pose}: {problem}")
    first = poses[next(iter(poses))]
    return out + move_problems(model, data, first if start is None else start, poses)


def describe(args, poses, limits, model, data):
    config, tcp = args.config, args.config["tcp"]
    null_target = "the posture at each base pose" if config["null_target"] is None else config["null_target"].tolist()
    print(f"\n  {args.activate}: ee_kp {config['ee_kp'].tolist()}, ee_kd {config['ee_kd'].tolist()}, "
          f"tool {config.get('tool', 'not set')}")
    print(f"  null_kp {config['null_kp'].tolist()}, null_kd {config['null_kd'].tolist()}, null target {null_target}")
    print(f"  TCP {tcp[:3, 3].tolist()} m from the flange, targets at {args.hz} Hz")
    print(f"  {len(poses) * args.repeats} blocks, {args.repeats} at each of {', '.join(poses)}, "
          f"about {len(poses) * args.repeats * 18 / 60:.1f} min")
    size = np.abs(np.concatenate(waves())).max(0)
    speed = np.abs(np.gradient(waves()[0], 1.0 / CONTROL_HZ, axis=0)).max(0)
    for name, q in poses.items():
        limit, scale = limits[name], limits[name]["scale"]
        print(f"  {name}: TCP at {np.round(kinematics(model, data, q, tcp)[0][:3, 3], 3).tolist()} m, "
              f"joints {np.round(q, 3).tolist()}")
        print(f"    steps up to {1000 * limit['step'][:3].max():.0f} mm and {1000 * limit['step'][3:].max():.0f} mrad "
              f"(torques up to {limit['step_torque'][:4].max():.0f} / {limit['step_torque'][4:].max():.1f} Nm)")
        print(f"    multisine and ramps up to {1000 * scale * size[:3].max():.0f} mm and "
              f"{1000 * scale * size[3:].max():.0f} mrad, at up to {scale * speed[:3].max():.2f} m/s and "
              f"{scale * speed[3:].max():.2f} rad/s (torques up to {limit['wave_torque'][:4].max():.0f} / "
              f"{limit['wave_torque'][4:].max():.1f} Nm, joint speeds up to {limit['wave_speed'].max():.2f} rad/s)"
              + (f", {100 * limit['shrunk']:.0f}% of their size for the clearance" if limit["shrunk"] < 1 else ""))


async def main() -> int:
    args = arguments(__doc__, "osc")
    config = args.config
    tcp, q_null = config["tcp"], config["null_target"]
    names = list(POSES if q_null is None else NULL_POSES)
    if any(name not in names for name in args.poses or []):
        print(f"\n  --poses takes {names} with this configuration\n")
        return 1

    # Plan with aiofranka's model, which carries the payload the robot compensates once connected.
    robot = None if args.plan else RobotInterface(args.ip)
    dynamics = mujoco.MjModel.from_xml_path(str(MODEL_PATH)) if robot is None else robot.model
    model, data = _collision_model(args.tool_length, args.tool_radius, args.floor, args.clearance)
    poses, trouble = base_poses(args.poses or names, tcp, q_null, model, data)
    limits = sizes(dynamics, poses, tcp, config["ee_kp"], config["ee_kd"], args.hz)
    shrink_waves(limits, poses, args.hz, tcp, q_null, model, data, args.clearance)
    blocks = plan(poses, {p: limits[p]["step"] for p in poses},
                  {p: [wave * limits[p]["scale"] for wave in waves()] for p in poses}, args.hz, args.repeats)
    describe(args, poses, limits, model, data)
    start = None if robot is None else robot.data.qpos[:7].copy()
    if not report(trouble + problems(blocks, poses, tcp, q_null, model, data, args.clearance, start), args):
        return 1
    if args.plan:
        return 0

    controller = NativeFrankaController(robot)
    if not args.no_check_tool:
        try:
            controller.check_tool(config)
        except RuntimeError as error:
            print(f"\n  {error} Here: --no-check-tool.\n")
            return 1
    print(f"  Payload the robot compensates: {robot.payload['mass']:.3f} kg at "
          f"{np.round(robot.payload['com'], 4).tolist()} m")
    if not args.yes:
        input(f"\n  The arm will move to {next(iter(poses))} and play {len(blocks)} blocks. "
              "Press Enter to start, Ctrl+C to cancel... ")
    print()
    meta = metadata(args, robot, controller, "osc", list(poses))
    played, null_targets = [], np.full((len(blocks), 7), np.nan)

    async def play_block(index, block):
        await move_to(controller, poses[block.pose])
        if q_null is not None:
            rest = null_rest(model, data, controller.state["qpos"], tcp, q_null)
            if rest > NULL_REST_TOLERANCE:
                raise Stop(f"the null space is not at rest at {block.pose} ({rest:.2f} rad toward the null target)")
        # At the base pose, where the null space is at rest, rather than at the null target.
        controller.activate(config, check_tool=False, check_null_target=False)
        null_targets[index] = controller.initial_qpos
        await asyncio.sleep(SETTLE)
        base = np.array(controller.ee_desired)
        await play(controller, "ee_desired", apply(base, block.offsets), RECORDED, played)

    stopped = await collect(controller, blocks, play_block, home=poses[next(iter(poses))])

    arrays = rows(played, blocks, RECORDED)
    arrays.update({name: np.tile(config[name], (len(blocks), 1)) for name in ("ee_kp", "ee_kd", "null_kp", "null_kd")})
    arrays.update(null_target=null_targets, rate_hz=np.full(len(blocks), args.hz),
                  pose=np.array([list(poses).index(b.pose) for b in blocks]), poses=np.array(list(poses.values())),
                  tcp=tcp)
    meta.update(ticks=len(arrays.get("time", ())), abort_reason=stopped, completed=stopped is None)
    path = output_path(args.out, "osc", not robot.real)
    save(path, arrays, meta)
    print(f"\n  Saved {path}" + (f"\n  Stopped: {stopped}" if stopped else ""))
    if "ee" in arrays:
        ee, ee_des = arrays["ee"], arrays["ee_des"]
        position = np.linalg.norm(ee_des[:, :3, 3] - ee[:, :3, 3], axis=1)
        angle = Rotation.from_matrix(ee_des[:, :3, :3] @ np.swapaxes(ee[:, :3, :3], 1, 2)).magnitude()
        summarize(arrays, meta, lambda r: f"{1000 * np.sqrt(np.mean(position[r] ** 2)):.1f} mm, "
                                   f"{1000 * np.sqrt(np.mean(angle[r] ** 2)):.1f} mrad")
    if robot.real and stopped is None:
        # A complete recording from the robot: list it in the configuration, where the fit finds it.
        add_recording(args.activate, {"path": path, "date": meta["date"], "tool": meta["tool"]}, controller=config)
        print(f"\n  Added it to the recordings of {args.activate}; fit it with:\n"
              f"    python examples/07_fit_osc_sysid.py --activate {args.activate} --physics_dt <s>")
    print()
    return 0 if stopped is None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
