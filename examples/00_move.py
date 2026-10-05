#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

from aiofranka import NativeFrankaController, RobotInterface


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move one joint on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    parser.add_argument("--joint", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.joint < 7:
        parser.error(f"--joint must be in [0, 6], got {args.joint}")

    controller = NativeFrankaController(RobotInterface(args.ip))
    await controller.start()
    try:
        target = controller.initial_qpos.copy()
        target[args.joint] += args.delta
        await controller.move(target)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
