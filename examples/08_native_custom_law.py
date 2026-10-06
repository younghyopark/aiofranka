#!/usr/bin/env python3
"""
Hold the arm with a control law written in Python and run by the native loop.

The controller runs the 1 kHz loop in C++. The law below is compiled with Numba
(pip install "aiofranka[native]") and called every millisecond without Python.
"""

from __future__ import annotations

import argparse

import numpy as np

from aiofranka import Controller, Robot, control_law


@control_law(params={"stiffness": 7, "damping": 7, "integral_gain": 7}, memory={"integral": 7})
def pid_hold(s, p, m, tau):
    """Joint PID: stiffness * error + integral_gain * integral of the error - damping * velocity."""
    error = p.q_desired - s.qpos
    m.integral[:] += error * s.dt
    tau[:] = p.stiffness * error + p.integral_gain * m.integral - p.damping * s.qvel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a custom control law on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    args = parser.parse_args()

    robot = Robot(args.ip)
    with Controller(robot) as controller:  # start() here, stop() at the end (or Ctrl+C)
        print("Moving to initial position...")
        controller.move()

        controller.switch(pid_hold)  # compiles the law; the loop keeps holding meanwhile
        controller.stiffness = np.ones(7) * 80.0
        controller.damping = np.ones(7) * 4.0
        controller.integral_gain = np.ones(7) * 2.0
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        while True:
            controller.set("q_desired", q0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
