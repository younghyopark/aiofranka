#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

import numpy as np

from aiofranka import Controller, Robot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream zero torque on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    args = parser.parse_args()

    robot = Robot(args.ip)
    with Controller(robot) as controller:  # start() here, stop() at the end (or Ctrl+C)
        print("Moving to initial position...")
        controller.move()

        time.sleep(3.0)
        print("Starting zero torque. The robot only compensates gravity; keep clear of it.")

        controller.switch("torque")
        controller.set_freq(10)
        zero_torque = np.zeros(7)
        while True:
            controller.set("torque", zero_torque)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
