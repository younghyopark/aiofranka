import asyncio
import unittest

import mujoco
import numpy as np

from aiofranka.controller import FrankaController
from aiofranka.payload import MODEL_PATH
from test_payload import simulated_robot


def tip(z):
    transform = np.eye(4)
    transform[2, 3] = z
    return transform


class SetTcpTest(unittest.TestCase):
    controller_cls = FrankaController  # test_native runs these with NativeFrankaController

    def setUp(self):
        self.robot = simulated_robot(mujoco.MjModel.from_xml_path(str(MODEL_PATH)))
        self.controller = self.controller_cls(self.robot)

    def test_tracks_the_new_tcp_where_it_is(self):
        self.controller.switch("osc")
        self.controller.set_tcp(tip(0.1))

        np.testing.assert_allclose(self.controller.control_transform, tip(0.1))
        np.testing.assert_allclose(self.controller.ee_desired, self.robot._ee() @ tip(0.1))
        # The OSC tracks the TCP, which is 10 cm out from the flange.
        offset = self.controller.ee_desired[:3, 3] - self.robot._ee()[:3, 3]
        self.assertAlmostEqual(np.linalg.norm(offset), 0.1)

    def test_takes_a_translation(self):
        self.controller.set_tcp([0.0, 0.0, 0.1034])

        np.testing.assert_allclose(self.controller.control_transform, tip(0.1034))

    def test_rejects_what_is_not_a_pose(self):
        sheared = tip(0.1)
        sheared[0, 1] = 0.5
        mirrored = np.diag([1.0, 1.0, -1.0, 1.0])
        for transform in ([0.0, 0.1], np.ones((3, 3)), sheared, mirrored):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                self.controller.set_tcp(transform)
        np.testing.assert_allclose(self.controller.control_transform, np.eye(4))

    def test_osc_moves_the_tcp_to_the_target(self):
        controller = self.controller

        async def run():
            await controller.start()
            try:
                controller.switch("osc")
                controller.ee_kp = np.full(6, 300.0)
                controller.ee_kd = 2 * np.sqrt(controller.ee_kp)
                controller.set_tcp(tip(0.1))
                target = controller.ee_desired.copy()
                target[:3, 3] += [0.03, -0.02, 0.0]
                with controller.state_lock:
                    controller.ee_desired = target
                await asyncio.sleep(1.5)
                return target, self.robot._ee() @ tip(0.1)
            finally:
                controller.running = False
                controller.task.cancel()

        target, tcp = asyncio.run(run())

        np.testing.assert_allclose(tcp[:3, 3], target[:3, 3], atol=0.003)


if __name__ == "__main__":
    unittest.main()
