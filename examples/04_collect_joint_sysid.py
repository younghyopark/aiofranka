#!/usr/bin/env python3
"""
Collect joint impedance data to identify the FR3 with 05_fit_joint_sysid.py.

With a joint impedance configuration (kp, kd, the policy rate as frequency, and the tool;
see aiofranka.config), it plays 13 s blocks of joint targets around three base poses, two
at each, and records every 1 kHz control cycle to examples/sysid_data/joint_sysid_<date>.npz.

    python examples/04_collect_joint_sysid.py 173.16.0.2 --activate configs/joint_impedance.yaml
    mjpython examples/04_collect_joint_sysid.py --activate configs/joint_impedance.yaml   # MuJoCo

Each block holds, steps all joints twice, and plays a multisine and ramps (see
sysid.py). Steps go up to 0.12 rad (joints 1-4) and 0.2 rad (5-7), less where kp
times the step would pass half the torque limit. Before moving, every target is checked
for 0.1 rad from the joint limits, and every target and move between poses for clearance
between the arm, a cylinder around the tool and the floor; --plan only checks. While
playing, only the robot's own limits apply: past them its reflexes stop it, and the rows so
far are saved. Ctrl+C holds the arm where it is. Joint 7 oscillated at kd 20 (stable at
12). Keep a hand on the enabling device.
"""

from __future__ import annotations

import numpy as np

from aiofranka import Controller, Robot
from aiofranka.config import add_recording
from aiofranka.payload import _collision_model
from sysid import (FIELDS, TORQUE_LIMIT, arguments, collect, joint_problem, metadata, move_problems,
                   move_to, multisine, output_path, plan, play, ramps, report, rows, save, summarize)

POSES = {
    "home": [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854],
    "left": [0.6, -0.3, 0.2, -2.2, 0.0, 1.9, -0.2],
    "right": [-0.6, 0.2, -0.2, -1.6, 0.0, 1.8, -1.4],
}
STEP_MAX = np.array([0.12] * 4 + [0.2] * 3)  # rad
MULTISINE_PEAK = np.array([0.12] * 4 + [0.2] * 3)  # rad
MULTISINE_SPEED = np.array([0.4] * 4 + [0.8] * 3)  # rad/s
RAMP = np.full(7, 0.06)  # rad, reached in 0.75 s

# What every cycle records, by its name in the file.
RECORDED = FIELDS | {"q_des": "q_desired"}


def problems(blocks, poses, model, data, clearance, start):
    """What is wrong with the targets, and with the moves from start (the arm's pose, if known)."""
    out = []
    for pose, q0 in poses.items():
        targets = np.unique(np.concatenate([q0 + b.offsets for b in blocks if b.pose == pose]), axis=0)
        problem = next(filter(None, (joint_problem(model, data, q, clearance) for q in targets)), None)
        if problem:
            out.append(f"{pose}: {problem}")
    first = poses[next(iter(poses))]
    return out + move_problems(model, data, first if start is None else start, poses)


def main() -> int:
    args = arguments(__doc__, "impedance")
    kp, kd = args.config["kp"], args.config["kd"]
    poses = {name: np.array(POSES[name]) for name in args.poses or POSES}
    step = np.minimum(STEP_MAX, 0.5 * TORQUE_LIMIT / kp)
    waves = (multisine(MULTISINE_PEAK, MULTISINE_SPEED), ramps(RAMP))
    blocks = plan(poses, {p: step for p in poses}, {p: waves for p in poses}, args.hz, args.repeats)

    robot = None if args.plan else Robot(args.ip)
    model, data = _collision_model(args.tool_length, args.tool_radius, args.floor, args.clearance)
    print(f"\n  {args.activate}: kp {kp.tolist()}, kd {kd.tolist()}, targets at {args.hz} Hz, "
          f"tool {args.config.get('tool', 'not set')}")
    print(f"  {len(blocks)} blocks, {args.repeats} at each of {', '.join(poses)}, about "
          f"{len(blocks) * 16 / 60:.1f} min; steps up to {np.round(step, 3).tolist()} rad")
    start = None if robot is None else robot.state["qpos"].copy()
    if not report(problems(blocks, poses, model, data, args.clearance, start), args):
        return 1
    if args.plan:
        return 0

    controller = Controller(robot)
    if not args.no_check_tool:
        try:
            controller.check_tool(args.config)
        except RuntimeError as error:
            print(f"\n  {error} Here: --no-check-tool.\n")
            return 1
    print(f"  Payload the robot compensates: {robot.payload['mass']:.3f} kg at "
          f"{np.round(robot.payload['com'], 4).tolist()} m")
    if not args.yes:
        input(f"\n  The arm will move to {next(iter(poses))} and play {len(blocks)} blocks. "
              "Press Enter to start, Ctrl+C to cancel... ")
    print()
    meta = metadata(args, robot, controller, "impedance", list(poses))
    played = []

    def play_block(index, block):
        q0 = poses[block.pose]
        move_to(controller, q0)
        controller.activate(args.config, check_tool=False)  # checked above
        play(controller, "q_desired", q0 + block.offsets, RECORDED, played)

    stopped = collect(controller, blocks, play_block, home=poses[next(iter(poses))])

    arrays = rows(played, blocks, RECORDED)
    arrays.update(kp=np.tile(kp, (len(blocks), 1)), kd=np.tile(kd, (len(blocks), 1)),
                  rate_hz=np.full(len(blocks), args.hz), pose=np.array([list(poses).index(b.pose) for b in blocks]),
                  poses=np.array(list(poses.values())))
    meta.update(ticks=len(arrays.get("time", ())), abort_reason=stopped, completed=stopped is None)
    path = output_path(args.out, "joint", not robot.real)
    save(path, arrays, meta)
    print(f"\n  Saved {path}" + (f"\n  Stopped: {stopped}" if stopped else ""))
    error = arrays.get("q_des", np.zeros((0, 7))) - arrays.get("q", np.zeros((0, 7)))
    summarize(arrays, meta, lambda rows: f"{1000 * np.sqrt(np.mean(error[rows] ** 2)):.1f} mrad")
    if robot.real and stopped is None:
        # A complete recording from the robot: list it in the configuration, where the fit finds it.
        add_recording(args.activate, {"path": path, "date": meta["date"], "tool": meta["tool"]},
                      controller=args.config)
        print(f"\n  Added it to the recordings of {args.activate}; fit it with:\n"
              f"    python examples/05_fit_joint_sysid.py --activate {args.activate} --physics_dt <s>")
    print()
    return 0 if stopped is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
