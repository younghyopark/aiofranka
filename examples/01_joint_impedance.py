#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aiofranka import NativeFrankaController, RobotInterface


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run joint impedance on the robot at IP, or omit IP to run MuJoCo."
    )
    parser.add_argument("ip", nargs="?")
    args = parser.parse_args()

    controller = NativeFrankaController(RobotInterface(args.ip))
    await controller.start()
    try:
        print("Moving to initial position...")
        await controller.move()

        controller.switch("impedance")
        controller.kp = np.ones(7) * 80.0
        controller.kd = np.ones(7) * 4.0
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        while True:
            await controller.set("q_desired", q0)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
