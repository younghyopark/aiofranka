#!/usr/bin/env python3

from __future__ import annotations

import argparse

from aiofranka import Controller, Robot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move one joint on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    parser.add_argument("--joint", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.joint < 7:
        parser.error(f"--joint must be in [0, 6], got {args.joint}")

    robot = Robot(args.ip)
    with Controller(robot) as controller:  # start() here, stop() at the end
        target = robot.state["qpos"].copy()
        target[args.joint] += args.delta
        controller.move(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
