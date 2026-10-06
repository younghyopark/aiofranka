#!/usr/bin/env python3
"""
Fit a MuJoCo FR3 under operational space control to a 06_collect_osc_sysid.py recording
with CMA-ES, at your simulation's physics step.

The simulation runs the way a policy's environment does: at each policy step it takes
the recorded TCP target, and for each of its decimation = (1 / hz) / physics_dt physics
steps it computes aiofranka's OSC in Python and applies it through fr3.xml's <motor>
actuators:

    tau = J^T Lambda (ee_kp * e - ee_kd * J dq) + (I - J^T Jbar^T) (null_kp * (q_null - q) - null_kd * dq)

with the TCP and null-space target of the recording, the 990 Nm/s rate limit and the
torque clip. Lambda and Jbar come from a mass matrix with fr3.xml's armature, as aiofranka
computes them on the robot. Starting from the recorded gains and fr3.xml's joint
parameters, it fits ee_kp, ee_kd, null_kp, null_kd and each joint's damping and friction
loss to the measured motion, replaying 2 s windows from the measured state at their start.
The armature stays at fr3.xml's, like the link inertias: fitted, it mostly trades off
against the gains instead of measuring the joints. --fixed chooses what stays at its start
value; with the armature fitted, --mass_matrix plant gives the OSC the fitted armature too,
for an environment that computes the OSC from its simulated model's mass matrix. The error
weighs the joints by --joint_weight (0.8) and the TCP by the rest, half position and half
rotation:

    sqrt(joint_weight * joints^2 + (1 - joint_weight) * (tcp_position^2 + tcp_rotation^2) / 2)

each an RMS over the window samples (the joints per joint) relative to fr3.xml's on the
training windows, so fr3.xml with the recorded gains scores 1. The windows at one pose are
held out to check the fit.

    pip install mjbatch   # batched MuJoCo; it pins its own mujoco version
    python examples/07_fit_osc_sysid.py --activate configs/<name>.yaml --physics_dt 0.002
    python examples/07_fit_osc_sysid.py --traj examples/sysid_data/osc_sysid_<date>.npz --physics_dt 0.002

With --activate, it fits the configuration's latest recording (06_collect_osc_sysid.py lists the
complete recordings it makes on the robot there); with --traj, that recording. It adds the
fit to the sim section of the configuration the recording was collected with (or
--activate), as the entry for this physics_dt (see aiofranka.config).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from mjbatch import Batch

from aiofranka.config import latest_recording, load_config, same_controller, save_sim
from aiofranka.payload import MODEL_PATH
from aiofranka.robot import link_inertial, merge_payload

CONTROL_HZ = 1000
GAINS = ("ee_kp", "ee_kd", "null_kp", "null_kd")
PHYSICAL = ("armature", "damping", "frictionloss")
SIZES = {"ee_kp": 6, "ee_kd": 6, "null_kp": 7, "null_kd": 7, "armature": 7, "damping": 7, "frictionloss": 7}
BOUNDS = {"ee_kp": (0.1, 1e4), "ee_kd": (0.01, 1e3), "null_kp": (0.01, 1e3), "null_kd": (0.01, 1e2),
          "armature": (0.001, 2.0), "damping": (0.001, 20.0), "frictionloss": (0.001, 10.0)}


class Run:
    """A 06_collect_osc_sysid.py recording: one set of gains, null-space target, TCP and policy rate."""

    def __init__(self, path):
        data = np.load(path)
        self.meta = json.loads(str(data["meta"]))
        if self.meta.get("controller") != "osc":
            raise ValueError("Not an OSC recording; fit joint impedance recordings with 05_fit_joint_sysid.py")
        for key in ("time", "q", "dq", "ee_des", "tau_cmd", "tau_J_d", "pose", "tcp", "segment"):
            setattr(self, key, data[key])
        self.block = data["block"].astype(int)
        self.gains = {}
        for name in GAINS + ("rate_hz",):
            values = data[name]
            if len(np.unique(values, axis=0)) > 1:
                raise ValueError(f"The recording has several values of {name}; record one setting per file")
            self.gains[name] = values[0].astype(float)
        self.hz = int(self.gains.pop("rate_hz"))
        # Each block's null-space target: the configuration's, or the posture the block started in.
        self.null_targets = data["null_target"].astype(float)
        played = self.null_targets[np.unique(self.block)]
        self.null_target = played[0] if len(np.unique(played, axis=0)) == 1 else None
        self.torque_limit = np.array(self.meta["torque_limit"])
        self.rate_limit = self.meta["torque_rate_limit"]
        self.payload = self.meta["load"]["payload"]
        self.pose_names = self.meta["pose_names"]

    def windows(self, length, poses):
        """First ticks of back-to-back windows from the start of each block at the poses, without lost packets."""
        lost = np.zeros(len(self.time), bool)
        lost[1:] = np.abs(np.diff(self.time) - 1.0 / CONTROL_HZ) > 0.25e-3
        lost_so_far = np.cumsum(lost)
        starts = []
        for block in np.unique(self.block):
            if self.pose_names[self.pose[block]] not in poses:
                continue
            ticks = np.flatnonzero(self.block == block)
            for start in range(ticks[0], ticks[-1] + 1 - length, length):
                if lost_so_far[start + length] == lost_so_far[start]:
                    starts.append(start)
        return np.array(starts, int)


def build_model(run, physics_dt):
    """
    fr3.xml in free space, with the payload the robot compensated merged into its last
    link as RobotInterface.sync_payload() merges it into the model of aiofranka's controllers.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = physics_dt
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    payload = run.payload
    merge_payload(model, link_inertial(model), payload["mass"], payload["com"], payload["inertia"])
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def rotvec(matrix):
    """Rotation vectors of rotation matrices (..., 3, 3) -> (..., 3), for angles below pi."""
    vee = 0.5 * np.stack([matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0],
                          matrix[..., 1, 0] - matrix[..., 0, 1]], -1)
    sin = np.linalg.norm(vee, axis=-1, keepdims=True)
    cos = 0.5 * (np.trace(matrix, axis1=-2, axis2=-1)[..., None] - 1.0)
    angle = np.arctan2(sin, cos)
    return vee * np.where(sin > 1e-12, angle / np.where(sin > 1e-12, sin, 1.0), 1.0)


class Simulator:
    """Batched closed-loop replays of windows of a run."""

    def __init__(self, run, physics_dt, mass_matrix="nominal", threads=0):
        self.run, self.physics_dt, self.mass_matrix, self.threads = run, physics_dt, mass_matrix, threads
        self.period = CONTROL_HZ // run.hz  # ticks per policy step
        self.ticks = round(physics_dt * CONTROL_HZ)  # ticks per physics step
        if not math.isclose(self.ticks / CONTROL_HZ, physics_dt) or self.period % self.ticks:
            raise ValueError(f"physics_dt must divide the policy period, 1 / {run.hz} s, in whole milliseconds")
        self.decimation = self.period // self.ticks
        self.every = math.lcm(self.ticks, 10)  # ticks between position samples
        self.model = build_model(run, physics_dt)
        self.site = self.model.site("attachment_site").id
        self.nominal_armature = self.model.dof_armature[:7].copy()
        # The mass matrix is stored as its lower triangle, row by row.
        self.rows = np.repeat(np.arange(7), self.model.M_rownnz[:7])
        self.cols = self.model.M_colind[: len(self.rows)]
        self.batches = {}

    def window_length(self, seconds):
        """Ticks closest to seconds that hold whole policy steps and samples."""
        unit = math.lcm(self.period, self.every)
        return max(1, round(seconds * CONTROL_HZ / unit)) * unit

    def start_values(self):
        """Recorded gains and fr3.xml's joint parameters."""
        values = {name: self.run.gains[name].copy() for name in GAINS}
        values.update({name: getattr(self.model, f"dof_{name}")[:7].copy() for name in PHYSICAL})
        return values

    def _batch(self, count):
        if count not in self.batches:
            # forward=True keeps the kinematics and mass matrix current with the state, as on the robot.
            batch = Batch(self.model, count, self.threads, forward=True)
            fields = {name: batch.bind(name) for name in
                      ("qpos", "qvel", "ctrl", "M", "xanchor", "xaxis", "site_xpos", "site_xmat")}
            expanded = {name: batch.expand(f"dof_{name}") for name in PHYSICAL}
            self.batches = {count: (batch, fields, expanded)}  # keep one, they are large
        return self.batches[count]

    def torque(self, fields, goal, gains, q_null, armature, previous):
        """aiofranka's OSC (FrankaController._osc_step), for every simulation at once."""
        run = self.run
        q, dq = fields["qpos"][:, :7], fields["qvel"][:, :7]
        flange_pos = fields["site_xpos"][:, self.site]
        flange_rot = fields["site_xmat"][:, self.site].reshape(-1, 3, 3)
        # Jacobian of the flange: hinge axes and anchors.
        axis, anchor = fields["xaxis"][:, :7], fields["xanchor"][:, :7]
        jac = np.concatenate([np.cross(axis, flange_pos[:, None] - anchor), axis], -1).transpose(0, 2, 1)
        # The TCP and its Jacobian.
        ee_rot = flange_rot @ run.tcp[:3, :3]
        offset = flange_rot @ run.tcp[:3, 3]
        ee_pos = flange_pos + offset
        jac[:, :3] += np.cross(jac[:, 3:].transpose(0, 2, 1), offset[:, None]).transpose(0, 2, 1)
        mass = np.zeros((len(q), 7, 7))
        mass[:, self.rows, self.cols] = fields["M"][:, : len(self.rows)]
        mass[:, self.cols, self.rows] = fields["M"][:, : len(self.rows)]
        if self.mass_matrix == "nominal":
            mass[:, np.arange(7), np.arange(7)] += self.nominal_armature - armature

        twist = np.concatenate([goal[:, :3, 3] - ee_pos, rotvec(goal[:, :3, :3] @ ee_rot.transpose(0, 2, 1))], -1)
        ee_vel = np.einsum("nij,nj->ni", jac, dq)
        minv = np.linalg.inv(mass)
        jac_t = jac.transpose(0, 2, 1)
        mx = np.linalg.inv(jac @ minv @ jac_t)
        feedback = np.einsum("nij,nj->ni", jac_t @ mx, gains["ee_kp"] * twist - gains["ee_kd"] * ee_vel)
        ddq = gains["null_kp"] * (q_null - q) - gains["null_kd"] * dq
        jbar = minv @ jac_t @ mx
        null = ddq - np.einsum("nij,nj->ni", jac_t, np.einsum("nji,nj->ni", jbar, ddq))
        tau = feedback + null
        step_limit = run.rate_limit * self.physics_dt
        tau = previous + np.clip(tau - previous, -step_limit, step_limit)
        return np.clip(tau, -run.torque_limit, run.torque_limit)

    def _load(self, params, starts):
        run = self.run
        count_params, count_windows = len(params["ee_kp"]), len(starts)
        batch, fields, expanded = self._batch(count_params * count_windows)
        per_sim = {name: np.repeat(params[name], count_windows, axis=0) for name in SIZES}
        for name in PHYSICAL:
            expanded[name][:, :7] = per_sim[name]
        batch.set_const()
        ticks = np.tile(starts, count_params)
        fields["qpos"][:] = run.q[ticks]
        fields["qvel"][:] = run.dq[ticks]
        per_sim["null_target"] = run.null_targets[run.block[ticks]]
        batch.forward()
        return batch, fields, per_sim, ticks

    def rollout(self, params, starts, length):
        """
        Simulated motion every self.every ticks of each window, for each parameter set.

        Args:
            params (dict): ee_kp, ee_kd (P, 6) and null_kp, null_kd, armature, damping, frictionloss (P, 7)
            starts (np.ndarray): First ticks of the windows (W,)
            length (int): Ticks per window, from window_length()

        Returns:
            dict: Joint positions q (P, W, S, 7), TCP positions pos (P, W, S, 3) and rotations
                rot (P, W, S, 3, 3), with S = length // self.every
        """
        run = self.run
        batch, fields, per_sim, ticks = self._load(params, starts)
        previous = run.tau_J_d[ticks].copy()  # the robot's last command at the window start
        samples = length // self.every
        out = {"q": np.empty((len(ticks), samples, 7)), "pos": np.empty((len(ticks), samples, 3)),
               "rot": np.empty((len(ticks), samples, 3, 3))}
        elapsed = 0  # ticks since the window start
        for _step in range(length // self.period):
            for _ in range(self.decimation):
                # The target the robot's controller had then: the policy's action, held
                # from when it was set, at whatever tick that was.
                goal = run.ee_des[ticks + elapsed]
                tau = self.torque(fields, goal, per_sim, per_sim["null_target"], per_sim["armature"], previous)
                fields["ctrl"][:] = tau
                previous = tau
                batch.step()
                elapsed += self.ticks
                if elapsed % self.every == 0:
                    sample = elapsed // self.every - 1
                    flange_rot = fields["site_xmat"][:, self.site].reshape(-1, 3, 3)
                    out["q"][:, sample] = fields["qpos"][:, :7]
                    out["pos"][:, sample] = fields["site_xpos"][:, self.site] + flange_rot @ run.tcp[:3, 3]
                    out["rot"][:, sample] = flange_rot @ run.tcp[:3, :3]
        shape = (len(params["ee_kp"]), len(starts), samples)
        return {key: value.reshape(shape + value.shape[2:]) for key, value in out.items()}

    def replay_error(self, count=2000, seed=0):
        """Largest difference between the torques sent and those recomputed from the log [Nm]."""
        run = self.run
        rng = np.random.default_rng(seed)
        ticks = np.sort(rng.choice(len(run.time), size=min(count, len(run.time)), replace=False))
        start = self.start_values()
        batch, fields, per_sim, _ = self._load({k: v[None] for k, v in start.items()}, ticks)
        saved, self.physics_dt = self.physics_dt, 1.0 / CONTROL_HZ  # the robot's rate limit per tick
        try:
            tau = self.torque(fields, run.ee_des[ticks], per_sim, per_sim["null_target"], per_sim["armature"],
                              run.tau_J_d[ticks])
        finally:
            self.physics_dt = saved
        return np.abs(tau - run.tau_cmd[ticks]).max()

    def measured(self, starts, length):
        """Measured motion at the samples of rollout(), without the parameter axis."""
        q = self.run.q[starts[:, None] + np.arange(self.every, length + 1, self.every)[None]]
        pos, rot = self.tcp_poses(q)
        return {"q": q, "pos": pos, "rot": rot}

    def tcp_poses(self, q):
        """TCP positions (..., 3) and rotations (..., 3, 3) at joint positions q (..., 7)."""
        data = mujoco.MjData(self.model)
        pos, rot = np.empty(q.shape[:-1] + (3,)), np.empty(q.shape[:-1] + (3, 3))
        for index in np.ndindex(q.shape[:-1]):
            data.qpos[:7] = q[index]
            mujoco.mj_kinematics(self.model, data)
            flange_rot = data.site_xmat[self.site].reshape(3, 3)
            pos[index] = data.site_xpos[self.site] + flange_rot @ self.run.tcp[:3, 3]
            rot[index] = flange_rot @ self.run.tcp[:3, :3]
        return pos, rot


def errors(simulated, measured):
    """
    RMS errors of each parameter set (P,): joints [rad, per joint], TCP position [m] and
    TCP rotation [rad].
    """
    q = np.nan_to_num(simulated["q"] - measured["q"][None], nan=10.0)
    pos = np.nan_to_num(simulated["pos"] - measured["pos"][None], nan=10.0)
    rot = np.nan_to_num(rotvec(simulated["rot"] @ np.swapaxes(measured["rot"], -1, -2)[None]), nan=10.0)
    return {"joints": np.sqrt(np.mean(q ** 2, axis=(1, 2, 3))),
            "tcp_position": np.sqrt(np.mean(np.sum(pos ** 2, -1), axis=(1, 2))),
            "tcp_rotation": np.sqrt(np.mean(np.sum(rot ** 2, -1), axis=(1, 2)))}


# The smallest errors the fit's error is relative to, where fr3.xml already matches [m, rad, rad].
SCALE_FLOOR = {"tcp_position": 1e-4, "tcp_rotation": 1e-3, "joints": 1e-4}


def cost(error, weights, scale):
    """
    The fit's error (P,): the joints weighted by weights["joints"] and the TCP by the rest,
    half position and half rotation, each relative to scale (fr3.xml's), so fr3.xml scores 1.
    """
    tcp = 0.5 * ((error["tcp_position"] / scale["tcp_position"]) ** 2
                 + (error["tcp_rotation"] / scale["tcp_rotation"]) ** 2)
    joints = (error["joints"] / scale["joints"]) ** 2
    return np.sqrt((1.0 - weights["joints"]) * tcp + weights["joints"] * joints)


class CMA:
    """Plain CMA-ES (Hansen's tutorial defaults) minimizing over R^n from a mean of 0."""

    def __init__(self, n, popsize, sigma, seed):
        self.n, self.lam, self.mu = n, popsize, popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean, self.sigma = np.zeros(n), sigma
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.generation, self.rng = 0, np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, x, cost):
        best = x[np.argsort(cost)[: self.mu]]
        old, self.mean = self.mean, self.weights @ best
        y = (self.mean - old) / self.sigma
        inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt @ y
        self.generation += 1
        norm = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
        hsig = norm / self.chi < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        steps = (best - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (steps.T * self.weights) @ steps)
        self.sigma *= math.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chi - 1))
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


def decode(x, base, free):
    """
    Parameter sets from log offsets of the free parameters to their base values, clipped to
    BOUNDS, and the others at their base values: (P, n) -> dict of (P, size).
    """
    x = np.atleast_2d(x)
    out, i = {}, 0
    for name, size in SIZES.items():
        if name not in free:
            out[name] = np.repeat(base[name][None], len(x), axis=0)
            continue
        low, high = BOUNDS[name]
        out[name] = np.clip(np.maximum(base[name], low) * np.exp(x[:, i:i + size]), low, high)
        i += size
    return out


def fit(sim, starts, length, base, free, weights, generations, popsize, sigma, seed):
    """
    CMA-ES over log offsets of the free parameters to base; returns the best parameters,
    their error (relative to base's) and the scale it is relative to (base's errors on the
    windows).
    """
    measured = sim.measured(starts, length)
    cma = CMA(sum(SIZES[name] for name in free), popsize, sigma, seed)
    best_x = np.zeros(cma.n)
    start = errors(sim.rollout(decode(best_x, base, free), starts, length), measured)
    scale = {key: max(float(value[0]), SCALE_FLOOR[key]) for key, value in start.items()}
    best_cost = cost(start, weights, scale)[0]
    print(f"  generation   0: error {best_cost:.3f} (fr3.xml: TCP {1000 * scale['tcp_position']:.2f} mm, "
          f"{1000 * scale['tcp_rotation']:.1f} mrad, joints {1000 * scale['joints']:.2f} mrad)", flush=True)
    t0 = time.perf_counter()
    for generation in range(1, generations + 1):
        x = cma.ask()
        costs = cost(errors(sim.rollout(decode(x, base, free), starts, length), measured), weights, scale)
        cma.tell(x, costs)
        if costs.min() < best_cost:
            best_cost, best_x = costs.min(), x[costs.argmin()].copy()
        progress(generation, generations, best_cost, costs.min(), cma.sigma, t0, "")
    return {name: value[0] for name, value in decode(best_x, base, free).items()}, best_cost, scale


def progress(generation, generations, best, current, sigma, started, unit):
    """One line per generation: the best error so far and this generation's, the step size and the time left."""
    elapsed = time.perf_counter() - started
    left = elapsed / generation * (generations - generation)
    print(f"  generation {generation:3d}/{generations}: best {best:.3f}{unit}, this generation {current:.3f}{unit}, "
          f"sigma {sigma:.3f}, {elapsed / generation:.1f} s each, {left / 60:.1f} min left", flush=True)


def evaluate(sim, params, starts, length, weights, scale):
    """The errors of one parameter set over the windows [m, rad], and the fit's (relative) error."""
    if len(starts) == 0:
        return {key: float("nan") for key in ("error", "tcp_position", "tcp_rotation", "joints")}
    error = errors(sim.rollout({k: v[None] for k, v in params.items()}, starts, length),
                   sim.measured(starts, length))
    return {"error": float(cost(error, weights, scale)[0]), **{key: float(value[0]) for key, value in error.items()}}


def plot_windows(run, length, holdout, poses):
    """A steps window and a multisine window of the held-out pose, then of a training pose, labeled."""
    multisine = run.meta["segments"].index("multisine")
    out = []
    for pose in ([holdout] if holdout in poses else []) + [p for p in poses if p != holdout]:
        starts = run.windows(length, [pose])
        if len(starts) == 0 or len(out) == 4:
            continue
        starts = starts[run.block[starts] == run.block[starts[0]]]  # the first block at the pose
        mixed = max(starts, key=lambda start: np.mean(run.segment[start:start + length] == multisine))
        tag = "held out" if pose == holdout else "train"
        out += [(starts[0], f"{tag}, {pose}: steps"), (mixed, f"{tag}, {pose}: multisine")]
    return out


def pyplot():
    """matplotlib.pyplot without a display, or None if it is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  pip install matplotlib to save plots of the fit")
        return None
    return plt


def draw(plt, windows, t, rows, path, title):
    """
    One column per window and one row per channel: rows is a list of (ylabel, {line: (values
    (windows, samples), style)}).
    """
    fig, axes = plt.subplots(len(rows), len(windows), figsize=(3.4 * len(windows) + 1.2, 1.7 * len(rows) + 1),
                             sharex=True, squeeze=False)
    for row, (ylabel, lines) in enumerate(rows):
        for col, (_, label) in enumerate(windows):
            ax = axes[row, col]
            for name, (values, style) in lines.items():
                ax.plot(t, values[col], label=name, **style)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time in the window [s]", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


STYLES = {"target": dict(color="0.65", lw=1.0), "robot": dict(color="k", lw=1.8),
          "fr3.xml": dict(color="tab:red", lw=1.1, ls="--"), "fit": dict(color="tab:blue", lw=1.2)}


def plot_fit(sim, run, curves, windows, length, paths, title):
    """
    Save the response in the windows, in task space (TCP position and rotation) and in joint
    space: target, robot and each simulated parameter set of curves ({label: params}),
    relative to the robot's pose at the window start. Returns the files written.
    """
    plt = pyplot()
    if plt is None:
        return []
    starts = np.array([start for start, _ in windows])
    ticks = starts[:, None] + np.arange(sim.every, length + 1, sim.every)[None]
    t = np.arange(1, ticks.shape[1] + 1) * sim.every / CONTROL_HZ
    measured = sim.measured(starts, length)
    motion = {"target": {"pos": run.ee_des[ticks][..., :3, 3], "rot": run.ee_des[ticks][..., :3, :3]},
              "robot": measured}
    motion.update({label: {key: value[0] for key, value in
                           sim.rollout({k: v[None] for k, v in params.items()}, starts, length).items()}
                   for label, params in curves.items()})
    start_pos, start_rot = sim.tcp_poses(run.q[starts])
    start_q = run.q[starts][:, None]
    task = []
    for i, axis in enumerate("xyz"):
        task.append((f"TCP {axis} [mm]", {name: (1000 * (m["pos"][..., i] - start_pos[:, None, i]), STYLES[name])
                                         for name, m in motion.items()}))
    for i, axis in enumerate("xyz"):
        task.append((f"TCP rotation {axis} [mrad]",
                     {name: (1000 * rotvec(m["rot"] @ start_rot[:, None].swapaxes(-1, -2))[..., i], STYLES[name])
                      for name, m in motion.items()}))
    joints = [(f"joint {j + 1} [mrad]", {name: (1000 * (m["q"][..., j] - start_q[..., j]), STYLES[name])
                                         for name, m in motion.items() if "q" in m}) for j in range(7)]
    return [draw(plt, windows, t, task, paths[0], title), draw(plt, windows, t, joints, paths[1], title)]


def configuration(run, path):
    """
    The configuration file to add the fit to, path or the recording's, if it describes the
    recorded controller; and the recorded controller.
    """
    if path is None:
        if run.meta.get("sim"):
            raise ValueError("The recording comes from MuJoCo: pass --activate to say which configuration "
                             "its fit goes to")
        if not run.meta.get("config_path"):
            raise ValueError("The recording does not name its configuration; pass --activate")
        path = run.meta["config_path"]
        if not Path(path).exists():
            raise ValueError(f"The recording's configuration {path} is not there; pass --activate")
    path = Path(path)
    config = load_config(path)
    recorded = run.meta.get("config") or {
        "mode": "osc", "frequency": run.hz, "tcp": run.tcp.tolist(),
        "null_target": "current" if run.null_target is None else run.null_target.tolist(),
        "tool": config.get("tool"), **{name: run.gains[name].tolist() for name in GAINS}}
    if not same_controller(config, recorded):
        raise ValueError(f"{path} no longer describes the controller of the recording")
    return path, recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].replace("\n", " "),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--activate", type=Path,
                        help="Configuration: fit its latest recording (without --traj) and add the fit to it")
    parser.add_argument("--traj", type=Path,
                        help="Recording of 06_collect_osc_sysid.py to fit (default: the latest of --activate)")
    parser.add_argument("--physics_dt", "--physics-dt", type=float, required=True,
                        help="Physics step of your simulation [s], a whole number of ms dividing the policy period")
    parser.add_argument("--fixed", nargs="*", choices=tuple(SIZES), default=["armature"], metavar="NAME",
                        help="Parameters kept at their start values, the recorded gains and fr3.xml's joint "
                             f"parameters: any of {', '.join(SIZES)}; --fixed alone fits all of them. Fitted, the "
                             "armature mostly trades off against the gains instead of measuring the joints")
    parser.add_argument("--mass_matrix", "--mass-matrix", choices=("nominal", "plant"), default="nominal",
                        help="With the armature fitted, the armature in the mass matrix of the OSC's Lambda and Jbar: "
                             "fr3.xml's, as aiofranka on the robot, or the simulated joints' (each candidate's while "
                             "fitting), for an environment that computes the OSC from its simulated model's mass matrix")
    parser.add_argument("--holdout", default="last",
                        help="Pose whose windows are left out of the fit: a pose name, last, or none")
    parser.add_argument("--joint_weight", "--joint-weight", type=float, default=0.8,
                        help="Share of the joint error in the fit; the TCP's (half position, half rotation) is "
                             "the rest. Each error counts relative to fr3.xml's on the training windows")
    parser.add_argument("--window", type=float, default=2.0, help="Window length [s]")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.3, help="Initial step size, in log of the parameters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="Simulation threads (0: all CPUs)")
    args = parser.parse_args()
    if not 0.0 <= args.joint_weight <= 1.0:
        parser.error("--joint_weight is a share between 0 and 1")
    free = tuple(name for name in SIZES if name not in args.fixed)
    fixed = [name for name in SIZES if name not in free]
    if not free:
        parser.error("--fixed leaves nothing to fit")

    if args.traj is None:
        if args.activate is None:
            parser.error("pass --activate (its latest recording is fitted) or --traj")
        try:
            args.traj = latest_recording(args.activate)
        except (OSError, ValueError) as problem:
            parser.error(str(problem))
        if args.traj is None:
            parser.error(f"{args.activate} lists no recordings yet: collect with "
                         f"06_collect_osc_sysid.py --activate {args.activate}, or pass --traj")
        print(f"\n  The latest recording of {args.activate}: {args.traj}")
    try:
        run = Run(args.traj)
        config_path, recorded_controller = configuration(run, args.activate)
    except (OSError, ValueError) as problem:
        parser.error(str(problem))
    sim = Simulator(run, args.physics_dt, args.mass_matrix, threads=args.threads)
    recorded = sorted({run.pose_names[run.pose[b]] for b in np.unique(run.block)}, key=run.pose_names.index)
    holdout = recorded[-1] if args.holdout == "last" and len(recorded) > 1 else args.holdout
    if holdout not in recorded + ["none", "last"]:
        parser.error(f"--holdout must be one of {recorded}, last or none")
    length = sim.window_length(args.window)
    train = run.windows(length, [p for p in recorded if p != holdout])
    test = run.windows(length, [holdout])

    print(f"\n  {args.traj.name}: {run.hz} Hz, TCP translation {run.tcp[:3, 3].tolist()} m")
    print(f"  Tool {run.meta.get('tool') or '(not read from Desk)'}: the model, and the OSC's mass matrix, carry the payload the robot "
          f"compensated, {run.payload['mass']:.3f} kg at {np.round(run.payload['com'], 4).tolist()} m")
    print("  " + ", ".join(f"{name} {run.gains[name].tolist()}" for name in GAINS))
    print(f"  null target {'the posture each block started in' if run.null_target is None else run.null_target.tolist()}")
    print(f"  Replay check: max |recomputed - sent torque| = {sim.replay_error():.1e} Nm")
    mass_matrix = f", {args.mass_matrix} mass matrix" if "armature" in free else ""
    print(f"  physics_dt {args.physics_dt * 1000:g} ms, decimation {sim.decimation}{mass_matrix}; "
          f"fitting {len(train)} windows of {length / CONTROL_HZ:g} s, holding out {len(test)} at {holdout}")
    print(f"  Fitting {', '.join(free)}" + (f"; keeping {', '.join(fixed)} at the start values" if fixed else "") + "\n")

    weights = {"joints": args.joint_weight, "tcp": 1.0 - args.joint_weight}
    print(f"  Error: {100 * weights['joints']:.0f}% joints, {100 * weights['tcp']:.0f}% TCP (half position, half "
          "rotation), each relative to fr3.xml's on the training windows\n")
    base = sim.start_values()
    params, _, scale = fit(sim, train, length, base, free, weights, args.generations, args.popsize, args.sigma,
                           args.seed)

    results = {label: {"train": evaluate(sim, p, train, length, weights, scale),
                       "held_out": evaluate(sim, p, test, length, weights, scale)}
               for label, p in (("start", base), ("fit", params))}
    print(f"\n  RMS                          error   TCP position [mm]   TCP rotation [mrad]   joints [mrad]")
    for label, result in results.items():
        for split, e in result.items():
            name = f"{label}, {split.replace('_', ' ')}" + (f" ({holdout})" if split == "held_out" else "")
            print(f"  {name:28s} {e['error']:5.3f}   {1000 * e['tcp_position']:17.2f}"
                  f"   {1000 * e['tcp_rotation']:19.2f}   {1000 * e['joints']:13.2f}")
    print("\n  axis   ee_kp              ee_kd")
    for i, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
        print(f"  {axis:4s}   " + "   ".join(f"{base[n][i]:7.4g} -> {params[n][i]:<7.4g}" for n in ("ee_kp", "ee_kd")))
    print("\n  joint  null_kp          null_kd          armature         damping          frictionloss")
    for j in range(7):
        print(f"  {j + 1}      " + "  ".join(f"{base[n][j]:6.3g} -> {params[n][j]:<6.3g}"
                                          for n in ("null_kp", "null_kd") + PHYSICAL))

    stem = args.traj.with_name(f"{args.traj.stem}_fit_{1000 * args.physics_dt:g}ms")
    before, after = results["start"]["held_out"], results["fit"]["held_out"]
    plots = plot_fit(sim, run, {"fr3.xml": base, "fit": params}, plot_windows(run, length, holdout, recorded),
                     length, [stem.with_name(f"{stem.name}_task.png"), stem.with_name(f"{stem.name}_joints.png")],
                     f"{args.traj.name}, physics_dt {1000 * args.physics_dt:g} ms: held-out TCP "
                     f"{1000 * before['tcp_position']:.2f} -> {1000 * after['tcp_position']:.2f} mm, "
                     f"{1000 * before['tcp_rotation']:.1f} -> {1000 * after['tcp_rotation']:.1f} mrad, "
                     f"joints {1000 * before['joints']:.2f} -> {1000 * after['joints']:.2f} mrad")
    for plot in plots:
        print(f"\n  Saved the response curves: {plot}")
    units = {"error": 1, "tcp_position": 1000, "tcp_rotation": 1000, "joints": 1000}
    save_sim(config_path, {
        "physics_dt": args.physics_dt,
        **{name: params[name] for name in GAINS},
        "mass_matrix": args.mass_matrix,  # armature in the OSC's mass matrix: fr3.xml's (nominal) or the joints' (plant)
        "mass_matrix_armature": sim.nominal_armature if args.mass_matrix == "nominal" else params["armature"],
        **{name: params[name] for name in PHYSICAL},
        # The payload the fit assumed, merged into fr3_link7 as aiofranka.robot.merge_payload() does.
        "payload": {"mass": run.payload["mass"], "com": run.payload["com"], "inertia": run.payload["inertia"]},
        "fit": {
            "data": args.traj.name,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "holdout": holdout,
            "plant": "mujoco" if run.meta.get("sim") else "robot",
            "tool": run.meta.get("tool"),
            "weights": weights,
            "fixed": fixed,
            "relative_to_mm_mrad": {key: 1000 * value for key, value in scale.items()},
            "rms_mm_mrad": {label: {split: {key: units[key] * value for key, value in e.items()}
                                    for split, e in result.items()} for label, result in results.items()},
            **({"plots": [os.path.relpath(p.resolve(), config_path.resolve().parent) for p in plots]} if plots else {}),
        },
    }, controller=recorded_controller)
    print(f"\n  Added the fit for physics_dt {args.physics_dt:g} to {config_path}\n")


if __name__ == "__main__":
    main()
