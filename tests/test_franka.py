"""
aiofranka.Robot and aiofranka.Controller: the sync API on the native loop. Most tests drive
the fake robot of test_native, libfranka's readOnce() and writeOnce() on a simulated arm; the
others run the loop in simulation.
"""

import _thread
import contextlib
import io
import threading
import time
import unittest

import numpy as np

from aiofranka.franka import Controller, Robot
from test_native import MODEL, fake_robot, needs_native
from test_payload import MASS, model_with_tool, simulated_robot


def quietly():
    return contextlib.redirect_stdout(io.StringIO())


def controller_threads():
    return [thread for thread in threading.enumerate() if thread.name == "aiofranka-controller"]


@needs_native
class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.robot = Robot._wrap(fake_robot())
        self.world = self.robot._interface.robot.world  # the simulated arm behind the fake robot
        self.controller = Controller(self.robot)

    def tearDown(self):
        self.controller.stop()

    def start(self):
        with quietly():
            self.controller.start()

    def test_drives_the_arm_and_gives_it_back(self):
        controller, robot = self.controller, self.robot
        self.assertIsNone(robot.controller)
        self.start()
        self.assertTrue(controller.running)
        self.assertIs(robot.controller, controller)
        self.assertEqual(sorted(robot.state), ["ee", "jac", "last_torque", "mm", "qpos", "qvel"])
        np.testing.assert_allclose(robot.state["qpos"], self.world.qpos[:7], atol=1e-3)
        controller.stop()
        self.assertFalse(controller.running)
        self.assertIsNone(robot.controller)
        self.assertEqual(robot.state["qpos"].shape, (7,))  # read from the arm again
        self.assertEqual(controller_threads(), [])
        self.start()  # and again
        self.assertTrue(controller.running)

    def test_set_holds_the_rate_and_the_arm_follows(self):
        controller, robot = self.controller, self.robot
        self.start()
        controller.switch("impedance")
        controller.set_freq(100)
        target = robot.state["qpos"] + np.array([0.05, 0, 0, 0, 0, 0, 0])
        began = time.perf_counter()
        for _ in range(50):
            controller.set("q_desired", target)
        self.assertAlmostEqual(time.perf_counter() - began, 0.5, delta=0.05)
        np.testing.assert_allclose(controller.q_desired, target)
        time.sleep(0.5)
        self.assertAlmostEqual(self.world.qpos[0], target[0], delta=0.01)

    def test_the_loop_runs_while_the_caller_blocks(self):
        controller, robot = self.controller, self.robot
        self.start()
        controller.switch("impedance")
        controller.q_desired = robot.state["qpos"] + np.array([0.1, 0, 0, 0, 0, 0, 0])
        controller.loop_stats(reset=True)
        time.sleep(0.5)  # blocks this thread
        stats = controller.loop_stats(reset=True)
        self.assertGreater(stats["count"], 450)
        self.assertLess(stats["max"], 5e-3)
        self.assertGreater(robot.state["qpos"][0] - controller.initial_qpos[0], 0.05)

    def test_move_reaches_the_target_and_records_every_cycle(self):
        controller, robot = self.controller, self.robot
        self.start()
        target = robot.state["qpos"] + np.array([0, 0.1, 0, 0, 0, 0, 0])
        with controller.record(["time", "q", "q_desired"]) as recording, quietly():
            began = time.perf_counter()
            controller.move(target)
            elapsed = time.perf_counter() - began
        np.testing.assert_allclose(robot.state["qpos"], target, atol=controller.arrival_tolerance)
        data = recording.data()
        self.assertAlmostEqual(len(data["time"]), elapsed * 1000, delta=60)
        # The trajectory's end; move() returns once the arm is there, maybe before the loop
        # takes the exact target
        np.testing.assert_allclose(data["q_desired"][-1], target, atol=1e-4)

    def test_ctrl_c_stops_move_and_the_arm_holds(self):
        controller, robot = self.controller, self.robot
        self.start()
        target = robot.state["qpos"] + np.array([0.5, 0, 0, 0, 0, 0, 0])
        threading.Timer(0.5, _thread.interrupt_main).start()
        with quietly(), self.assertRaises(KeyboardInterrupt):
            controller.move(target)
        self.assertTrue(controller.running)
        time.sleep(0.3)
        held = robot.state["qpos"][0]
        time.sleep(0.3)
        self.assertAlmostEqual(robot.state["qpos"][0], held, delta=0.005)
        self.assertLess(held, target[0] - 0.2)  # stopped short of the target

    def test_one_controller_drives_a_robot_at_a_time(self):
        self.start()
        other = Controller(self.robot)
        self.addCleanup(other.stop)
        with self.assertRaisesRegex(RuntimeError, "Another Controller drives this robot"):
            other.start()
        self.assertIs(self.robot.controller, self.controller)
        self.controller.stop()
        with quietly():
            other.start()
        try:
            self.assertIs(self.robot.controller, other)
        finally:
            other.stop()

    def test_its_attributes_are_the_controllers(self):
        controller = self.controller
        controller.kp = np.full(7, 60.0)  # before start, then while it runs
        self.start()
        controller.kd = np.full(7, 5.0)
        np.testing.assert_array_equal(controller.kp, np.full(7, 60.0))
        np.testing.assert_array_equal(controller._engine._records["kd"][0], np.full(7, 5.0))
        controller.switch("osc")
        self.assertEqual(controller.type, "osc")
        with self.assertRaisesRegex(AttributeError, "robot.state"):
            controller.state
        with self.assertRaisesRegex(AttributeError, "no attribute 'kpp' to set"):
            controller.kpp = 1.0
        with self.assertRaisesRegex(TypeError, "aiofranka.Robot"):
            Controller(self.robot._interface)

    def test_set_load_needs_the_arm_without_a_controller(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, "without a controller"):
            self.robot.set_load(0.5, inertia=[1e-3] * 3)

    def test_set_and_move_need_the_loop(self):
        with self.assertRaisesRegex(RuntimeError, "start\\(\\) the controller first"):
            self.controller.move()
        with self.assertRaisesRegex(RuntimeError, "start\\(\\) the controller first"):
            self.controller.set("q_desired", np.zeros(7))

    def test_set_keeps_the_period_to_about_a_cycle(self):
        controller, robot = self.controller, self.robot
        self.start()
        controller.switch("impedance")
        controller.set_freq(50)
        q0 = robot.state["qpos"]
        step = np.array([1e-3, 0, 0, 0, 0, 0, 0])
        with controller.record(["q_desired"]) as recording:
            for i in range(60):
                controller.set("q_desired", q0 + (i % 2) * step)  # a new target every period
        targets = recording.data()["q_desired"]
        changes = np.flatnonzero(np.any(targets[1:] != targets[:-1], axis=1)) + 1
        periods = np.diff(changes)  # cycles between new targets: 20 at 50 Hz
        self.assertGreater(len(periods), 50)
        self.assertLessEqual(np.sum(np.abs(periods - 20) > 1), len(periods) // 10)

    def test_checks_the_tool_and_reports_the_payload(self):
        interface = self.robot._interface
        interface.tool, interface.tool_error = None, "Desk could not be read"
        config = {"mode": "impedance", "kp": 80, "kd": 4, "frequency": 50}
        self.controller.check_tool(config)  # without a tool, any passes
        with self.assertRaisesRegex(RuntimeError, "Desk could not be read"):
            self.controller.check_tool({**config, "tool": "gripper"})
        self.assertEqual(sorted(self.robot.payload), ["com", "inertia", "mass"])


@needs_native
class FailureTest(unittest.TestCase):
    def test_a_loop_error_is_raised_by_the_next_call(self):
        robot = Robot._wrap(fake_robot(fail_after=1300, fail_message="Reflex: fake"))
        controller = Controller(robot)
        self.addCleanup(controller.stop)
        errors = []
        controller.error_callback = errors.append
        with quietly():
            controller.start()
            time.sleep(0.6)  # the fake robot fails 1.3 s in
        self.assertFalse(controller.running)
        self.assertEqual(errors, ["Reflex: fake"])
        self.assertEqual(controller.error, "Reflex: fake")
        self.assertEqual(robot.state["qpos"].shape, (7,))  # the arm is read again
        with self.assertRaisesRegex(RuntimeError, "The control loop stopped: Reflex: fake"):
            controller.set("q_desired", robot.state["qpos"])
        with self.assertRaisesRegex(RuntimeError, "Reflex: fake"):
            controller.kp = np.full(7, 50.0)
        self.assertIs(robot.controller, controller)  # until stop()
        controller.stop()
        self.assertIsNone(robot.controller)
        self.assertEqual(controller_threads(), [])

    def test_a_loop_that_fails_while_starting_raises_from_start(self):
        robot = Robot._wrap(fake_robot(fail_after=100, fail_message="Reflex: fake"))
        controller = Controller(robot)
        self.addCleanup(controller.stop)
        with quietly(), self.assertRaisesRegex(RuntimeError, "Reflex: fake"):
            controller.start()
        self.assertFalse(controller.running)
        self.assertIsNone(robot.controller)
        self.assertEqual(controller_threads(), [])


@needs_native
class SimulationTest(unittest.TestCase):
    def test_moves_the_simulated_arm(self):
        robot = Robot._wrap(simulated_robot(MODEL))
        self.assertFalse(robot.real)
        self.assertIsNone(robot.robot_state)
        controller = Controller(robot)
        with quietly():
            controller.start()
        try:
            target = robot.state["qpos"] + np.array([0, 0, 0, 0.1, 0, 0, 0])
            with quietly():
                controller.move(target)
            np.testing.assert_allclose(robot.state["qpos"], target, atol=controller.arrival_tolerance)
        finally:
            controller.stop()

    def test_identifies_the_tool_starting_and_stopping_the_loop(self):
        robot = Robot._wrap(simulated_robot(model_with_tool(gravcomp=0)))
        controller = Controller(robot)
        self.addCleanup(controller.stop)
        with quietly():
            estimate = controller.identify_payload(n_poses=3, speed=3.0, settle=0.2, duration=0.2,
                                                   verbose=False)
        self.assertAlmostEqual(estimate.mass, MASS, delta=0.01)
        self.assertFalse(controller.running)
        self.assertIsNone(robot.controller)
        self.assertEqual(controller_threads(), [])


if __name__ == "__main__":
    unittest.main()
