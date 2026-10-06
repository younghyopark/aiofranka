#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np

from aiofranka import Controller, Robot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OSC hold on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    args = parser.parse_args()

    robot = Robot(args.ip)
    with Controller(robot) as controller:  # start() here, stop() at the end (or Ctrl+C)
        print("Moving to initial position...")
        controller.move()

        controller.switch("osc")
        controller.ee_kp = np.ones(6) * 100.0
        controller.ee_kd = np.ones(6) * 20.0
        controller.null_kp = np.ones(7) * 9.0
        controller.null_kd = np.ones(7) * 6.0
        controller.set_freq(50)

        target = controller.initial_ee.copy()
        while True:
            controller.set("ee_desired", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
