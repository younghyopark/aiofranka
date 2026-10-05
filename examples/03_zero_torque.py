#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aiofranka import NativeFrankaController, RobotInterface


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream zero torque on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    args = parser.parse_args()

    controller = NativeFrankaController(RobotInterface(args.ip))
    await controller.start()
    try:
        print("Moving to initial position...")
        await controller.move()

        await asyncio.sleep(3.0)
        print("Starting zero torque. The robot only compensates gravity; keep clear of it.")

        controller.switch("torque")
        controller.set_freq(10)
        zero_torque = np.zeros(7)
        while True:
            await controller.set("torque", zero_torque)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
