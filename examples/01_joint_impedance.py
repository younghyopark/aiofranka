#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np

from aiofranka import Controller, Robot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run joint impedance on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    args = parser.parse_args()

    robot = Robot(args.ip)
    with Controller(robot) as controller:  # start() here, stop() at the end (or Ctrl+C)
        print("Moving to initial position...")
        controller.move()

        controller.switch("impedance")
        controller.kp = np.ones(7) * 80.0
        controller.kd = np.ones(7) * 4.0
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        while True:
            controller.set("q_desired", q0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
