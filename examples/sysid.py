"""
What the system identification collectors share, 04_collect_joint_sysid.py and
06_collect_osc_sysid.py.

They play 13 s blocks of targets around base poses at a configuration's policy rate, as a
policy would, and record every 1 kHz control cycle with NativeFrankaController.record():

    hold 0.5 s | 2 steps, 3 s | multisine 0.17-3 Hz, 6 s | slow ramps, 3 s | hold 0.5 s

The rows of all blocks go to one file, which 05_fit_joint_sysid.py and 07_fit_osc_sysid.py
fit. Each row has the target that cycle used, so the fits replay the targets exactly when
they were set.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiofranka.config import load_config, to_yaml
from aiofranka.payload import MODEL_PATH, _closest, _is_clear

CONTROL_HZ = 1000
SEGMENTS = ("hold", "steps", "multisine", "ramps")
HOLD = 0.5  # s at the start and the end of a block
STEP_COUNT, STEP_HOLD = 2, 0.75  # steps; s at each step's offset, then as long back
MULTISINE_S, MULTISINE_FMAX, FADE = 6.0, 3.0, 0.5  # s (the lowest harmonic's period), Hz, s of fade
RAMP_QUARTER = 0.75  # s from 0 to a ramp's peak
PLAN_LIMIT_MARGIN = 0.1  # rad between every target and the joint limits
TORQUE_LIMIT = np.array([87.0] * 4 + [12.0] * 3)  # FrankaController.torque_limit [Nm]

# Joint impedance for the moves between base poses, and for holding after a stop.
MOVE_KP = np.array([80.0] * 4 + [48.0] * 3)
MOVE_KD = np.array([8.0] * 4 + [6.0] * 3)
MOVE_SPEED = 0.5  # peak joint speed [rad/s]
SETTLE = 1.0  # s after a move, and after activating the configuration

# The fields of every cycle in the recording file, by the names record() takes them under.
FIELDS = {"time": "time", "q": "q", "dq": "dq", "tau_cmd": "tau", "tau_J_d": "last_torque",
          "tau_J": "tau_J", "success_rate": "control_command_success_rate"}


# ── The excitation ────────────────────────────────────────────────────────────

def ticks(seconds):
    return round(seconds * CONTROL_HZ)


def steps(size, rng):
    """STEP_COUNT steps of all axes at once, each to 0.5-1 times size, random signs, and back."""
    parts = []
    for _ in range(STEP_COUNT):
        offset = rng.choice([-1.0, 1.0], len(size)) * rng.uniform(0.5, 1.0, len(size)) * size
        parts += [np.tile(offset, (ticks(STEP_HOLD), 1)), np.zeros((ticks(STEP_HOLD), len(size)))]
    return np.concatenate(parts)


def multisine(size, speed):
    """
    Sums of sines from 1 / MULTISINE_S to MULTISINE_FMAX with other harmonics on each axis,
    amplitudes falling as 1 / frequency (about flat in velocity) and Schroeder phases,
    faded in and out, scaled to size or to speed, whichever is smaller.
    """
    axes = len(size)
    t = np.arange(ticks(MULTISINE_S)) / CONTROL_HZ
    fade = 0.5 - 0.5 * np.cos(np.pi * np.clip(np.minimum(t, MULTISINE_S - t) / FADE, 0.0, 1.0))
    offsets = np.zeros((len(t), axes))
    for axis in range(axes):
        k = np.arange(axis + 1, round(MULTISINE_FMAX * MULTISINE_S) + 1, axes)
        n = np.arange(len(k))
        phase = -np.pi * n * (n - 1) / len(k)
        shape = (np.sin(2 * np.pi * np.outer(t, k) / MULTISINE_S + phase) / k).sum(1) * fade
        peak_speed = np.abs(np.gradient(shape, 1.0 / CONTROL_HZ)).max()
        offsets[:, axis] = shape * min(size[axis] / np.abs(shape).max(), speed[axis] / peak_speed)
    return offsets


def ramps(size):
    """A triangle at constant speed, up and down to +/- size, alternating sign by axis."""
    u = np.arange(ticks(4 * RAMP_QUARTER)) / CONTROL_HZ / RAMP_QUARTER
    triangle = 1.0 - np.abs((u + 1.0) % 4.0 - 2.0)
    return np.outer(triangle, size * (-1.0) ** np.arange(len(size)))


def held(offsets, rate):
    """Offsets at 1 kHz held at the policy rate, as a policy's actions are."""
    period = CONTROL_HZ // rate
    return offsets[np.arange(len(offsets)) // period * period]


@dataclass
class Block:
    """One run of the excitation at a base pose, as offsets from it at the policy rate."""

    pose: str
    offsets: np.ndarray  # (policy steps, axes)
    segment: np.ndarray  # index into SEGMENTS of each policy step


def plan(poses, step_size, waves, rate, repeats, seed=0):
    """
    The blocks in the order they run, repeats at each pose with other steps.

    Args:
        poses (list): Names of the base poses
        step_size (dict): Largest step of each axis, by pose
        waves (dict): The multisine and the ramps at 1 kHz, by pose
        rate (int): Policy rate [Hz]
    """
    period = CONTROL_HZ // rate
    blocks = []
    for pose in poses:
        hold = np.zeros((ticks(HOLD), len(step_size[pose])))
        for _ in range(repeats):
            parts = [hold, steps(step_size[pose], np.random.default_rng([seed, len(blocks)])), *waves[pose], hold]
            segment = np.concatenate([np.full(len(p), i, np.int8) for i, p in zip((0, 1, 2, 3, 0), parts)])
            blocks.append(Block(pose, np.concatenate(parts)[::period], segment[::period]))
    return blocks


# ── Checks before moving ──────────────────────────────────────────────────────

def joint_problem(model, data, q, clearance):
    """What is wrong with joint positions q, or None."""
    lower = model.jnt_range[:7, 0] + PLAN_LIMIT_MARGIN
    upper = model.jnt_range[:7, 1] - PLAN_LIMIT_MARGIN
    if np.any(q < lower) or np.any(q > upper):
        return f"a joint is within {PLAN_LIMIT_MARGIN} rad of its limit at {np.round(q, 3).tolist()}"
    if not _is_clear(model, data, q):
        return f"{_closest(model, data, q, clearance)} at {np.round(q, 3).tolist()}"
    return None


def path_clear(model, data, start, end, step=0.02):
    """Whether the straight line in joint space from start to end keeps the clearance."""
    count = max(int(np.ceil(np.abs(end - start).max() / step)), 1)
    return all(_is_clear(model, data, start + s * (end - start)) for s in np.linspace(0.0, 1.0, count + 1))


def move_problems(model, data, start, poses):
    """Problems with the moves from start through the base poses and back to the first."""
    first = next(iter(poses))
    route = [("the current pose", start), *poses.items(), (first, poses[first])]
    return [f"the move from {a} to {b} is not clear" for (a, qa), (b, qb) in zip(route, route[1:])
            if a != b and not path_clear(model, data, qa, qb)]


# ── Collecting ────────────────────────────────────────────────────────────────

class Stop(Exception):
    """Stops the collection with a reason, holding the arm where it is."""


def hold(controller):
    """Hold the joint positions with joint impedance."""
    controller.switch("impedance")
    controller.kp, controller.kd = MOVE_KP, MOVE_KD


async def move_to(controller, target):
    """Move along a straight line in joint space, quintic in time, then settle."""
    hold(controller)
    start = np.array(controller.q_desired)
    duration = max(1.875 * np.abs(target - start).max() / MOVE_SPEED, 0.5)
    started = time.perf_counter()
    while (s := min((time.perf_counter() - started) / duration, 1.0)) < 1.0:
        controller.q_desired = start + s ** 3 * (10 - 15 * s + 6 * s ** 2) * (target - start)
        await asyncio.sleep(0.01)
    controller.q_desired = target
    await asyncio.sleep(SETTLE)


async def play(controller, attr, targets, fields, played):
    """
    Set attr to the targets one by one at the policy rate (activate() sets it for set()),
    recording every cycle. Adds the recording and the row each target was set at to played
    before it starts, so an interrupted block keeps its rows.
    """
    recording = controller.record(list(fields.values()))
    starts = []
    played.append((recording, starts))
    with recording:
        for target in targets:
            starts.append(recording.rows)
            await controller.set(attr, target)


async def collect(controller, blocks, play_block, home):
    """
    Start the loop, run play_block(index, block) for each block, move back to home and stop
    the loop. Ctrl+C, a Stop or an error holds the arm where it is first.

    Returns:
        str: Why it ended early, or None
    """
    errors = []
    controller.error_callback = errors.append  # the loop ended with an error, e.g. a reflex
    started = time.perf_counter()
    await controller.start()
    reason = None
    try:
        for index, block in enumerate(blocks):
            print(f"  [{index + 1}/{len(blocks)}] {block.pose}   ({(time.perf_counter() - started) / 60:.1f} min)")
            await play_block(index, block)
        print("  Moving back")
        await move_to(controller, home)
    except (asyncio.CancelledError, Exception) as stopped:
        if errors:
            reason = f"control loop error: {errors[0]}"
        else:
            if not isinstance(stopped, (asyncio.CancelledError, Stop)):
                traceback.print_exc()
            reason = "interrupted" if isinstance(stopped, asyncio.CancelledError) else str(stopped)
            hold(controller)
            await asyncio.sleep(0.5)
    finally:
        if errors:
            # The loop has ended, and its task, which stop() would wait for, exits the process.
            controller.robot.stop()
            if controller.task.done() and not controller.task.cancelled():
                controller.task.exception()  # seen, so asyncio does not print it again at exit
        else:
            await controller.stop()
    return reason


# ── The recording file ────────────────────────────────────────────────────────

def rows(played, blocks, fields):
    """The rows of all blocks in one table, with each row's block and segment."""
    columns = {name: [] for name in [*fields, "block", "segment"]}
    for index, (recording, starts) in enumerate(played):
        data = recording.data()
        count = len(data["time"])
        step = np.maximum(np.searchsorted(starts, np.arange(count), side="right") - 1, 0)
        for name, field in fields.items():
            columns[name].append(data[field])
        columns["block"].append(np.full(count, index, np.int16))
        columns["segment"].append(blocks[index].segment[step])
    return {name: np.concatenate(parts) for name, parts in columns.items() if parts}


def metadata(args, robot, controller, kind, pose_names):
    """What 05 and 07 need besides the rows, and how the recording was made."""
    root = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    try:
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "aiofranka", "examples"))
    except OSError:
        commit, dirty = "", True
    load = {"payload": {key: np.asarray(value).tolist() for key, value in robot.payload.items()}}
    if robot.robot_state is not None:  # what Desk's end-effector profile says
        load["robot_state"] = {key: np.asarray(getattr(robot.robot_state, key)).tolist() for key in
                               ("m_ee", "F_x_Cee", "I_ee", "m_load", "F_x_Cload", "I_load", "m_total",
                                "F_x_Ctotal", "I_total", "F_T_EE", "F_T_NE", "NE_T_EE")}
    return {
        "format": "aiofranka-sysid-3",
        "controller": kind,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "sim": not robot.real,
        "ip": args.ip,
        "control_hz": CONTROL_HZ,
        "torque_limit": controller.torque_limit.tolist(),
        "torque_rate_limit": controller.torque_diff_limit,
        "segments": SEGMENTS,
        "config": to_yaml(args.config),
        "config_path": str(args.activate.resolve()),
        "tool": robot.tool.name if robot.tool is not None else None,
        "tool_checked": not args.no_check_tool and robot.real,
        "hz": args.hz,
        "pose_names": pose_names,
        "load": load,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key != "config"},
        "aiofranka_commit": commit,
        "aiofranka_dirty": dirty,
        "fr3_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    }


def output_path(folder, kind, sim):
    """A new file in folder, named after the time."""
    stem = f"{kind}_sysid_{datetime.datetime.now():%Y%m%d_%H%M%S}{'_sim' if sim else ''}"
    path, suffix = folder / f"{stem}.npz", 2
    while path.exists():  # never overwrite another recording
        path, suffix = folder / f"{stem}-{suffix}.npz", suffix + 1
    return path


def save(path, arrays, meta):
    """Write the recording file, whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial.npz")
    np.savez(partial, **arrays, meta=np.array(json.dumps(meta, indent=1)))
    partial.replace(path)


def summarize(arrays, meta, tracking):
    """Print, per pose, tracking(rows), the peak torques and how often the rate limit and clip acted."""
    count = len(arrays.get("time", ()))
    if count == 0:
        return
    tau, previous = arrays["tau_cmd"], arrays["tau_J_d"]
    limit, rate_limit = np.array(meta["torque_limit"]), meta["torque_rate_limit"]
    rate_limited = (np.abs(tau - previous) >= 0.999 * rate_limit / CONTROL_HZ).any(1)
    clipped = (np.abs(tau) >= limit - 1e-9).any(1)
    print(f"\n  {count} cycles ({count / CONTROL_HZ / 60:.1f} min) in {len(np.unique(arrays['block']))} blocks")
    if not meta["sim"]:
        same_block = np.diff(arrays["block"]) == 0
        lost = np.round(np.diff(arrays["time"])[same_block] * CONTROL_HZ).astype(int) - 1
        print(f"  Robot states lost: {lost[lost > 0].sum()}, lowest success rate "
              f"{np.nanmin(arrays['success_rate']):.3f}")
    print("\n  pose     tracking (RMS)          max |tau| j1-4 / j5-7 [Nm]   rate-limited   clipped")
    pose = arrays["pose"][arrays["block"]]
    for index, name in enumerate(meta["pose_names"]):
        rows = pose == index
        if rows.any():
            peak = np.abs(tau[rows]).max(0)
            print(f"  {name:6s}   {tracking(rows):22s} {peak[:4].max():14.1f} / {peak[4:].max():4.1f}"
                  f"   {100 * rate_limited[rows].mean():11.1f}%   {100 * clipped[rows].mean():6.1f}%")


# ── The command line ──────────────────────────────────────────────────────────

def arguments(doc, kind):
    """The command line options of both collectors."""
    parser = argparse.ArgumentParser(description=doc.split("\n\n")[1].replace("\n", " "),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ip", nargs="?", help="Robot IP; omit it to run MuJoCo")
    parser.add_argument("--activate", type=Path, required=True,
                        help=f"{'Joint impedance' if kind == 'impedance' else 'OSC'} configuration "
                             "(YAML, see aiofranka.config)")
    parser.add_argument("--plan", action="store_true", help="Check the plan and exit")
    parser.add_argument("--poses", nargs="+", help="Base poses (default: all)")
    parser.add_argument("--repeats", type=int, default=2, help="Blocks at each pose")
    parser.add_argument("--no-check-tool", action="store_true",
                        help="Collect even if Desk's active end effector is not the configuration's tool "
                             "or cannot be read")
    parser.add_argument("--tool-length", type=float, default=0.2, help="Tool length from the flange [m]")
    parser.add_argument("--tool-radius", type=float, default=0.1, help="Tool radius [m]")
    parser.add_argument("--floor", type=float, default=0.0, help="Floor or table height in the base frame [m]")
    parser.add_argument("--clearance", type=float, default=0.05,
                        help="Clearance kept between the arm, the tool and the floor [m]")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "sysid_data",
                        help="Output folder")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not ask before moving")
    args = parser.parse_args()
    try:
        args.config = load_config(args.activate)
    except (OSError, ValueError) as error:
        parser.error(f"--activate {args.activate}: {error}")
    if args.config["mode"] != kind:
        other = "06_collect_osc_sysid.py" if kind == "impedance" else "04_collect_joint_sysid.py"
        parser.error(f"--activate needs mode {kind}, not {args.config['mode']}; use {other}")
    stiffness = "kp" if kind == "impedance" else "ee_kp"
    if np.any(args.config[stiffness] <= 0):
        parser.error(f"{stiffness} must be positive")
    args.hz = round(args.config["frequency"])
    if args.hz != args.config["frequency"] or CONTROL_HZ % args.hz:
        parser.error(f"The frequency must divide {CONTROL_HZ} Hz, not {args.config['frequency']}")
    return args


def report(problems, args):
    """Print the problems; returns whether there were none."""
    if problems:
        print("\n  Not safe to run:")
        for problem in problems:
            print(f"    {problem}")
        print()
        return False
    print(f"  Every target and move keeps {100 * args.clearance:.0f} cm of clearance "
          f"(tool {args.tool_length} m x {args.tool_radius} m, floor at {args.floor} m)")
    return True
