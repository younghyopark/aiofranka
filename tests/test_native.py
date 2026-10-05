"""
NativeFrankaController against FrankaController.

The parity tests step both controllers in lockstep from the same simulated state, so their
torques must agree to rounding. The fake-robot tests run the loop's real-robot path:
libfranka's readOnce() and writeOnce() on _FakeActiveControl, a C++ ActiveControl that
simulates the arm at 1 kHz.
"""

import asyncio
import contextlib
import inspect
import io
import os
import threading
import time
import types
import unittest

import mujoco
import numpy as np

import test_controller
from aiofranka.controller import FrankaController
from aiofranka.payload import MODEL_PATH
from test_payload import HOME, simulated_robot

try:
    from aiofranka.native import NativeFrankaController, _mujoco_functions, control_law, load_native

    load_native()
    NATIVE_ERROR = None
except ImportError as error:
    NATIVE_ERROR = str(error)

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_native = unittest.skipIf(NATIVE_ERROR is not None, f"native loop not built: {NATIVE_ERROR}")

MODEL = mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def robot_at(qpos):
    robot = simulated_robot(MODEL)
    robot.data.qpos[:] = qpos
    mujoco.mj_forward(robot.model, robot.data)
    return robot


def pose_about_z(angle):
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return rotation


class _StubPylibfrankaRobot:
    """pylibfranka.Robot for a fake robot: starts a _FakeActiveControl on a simulated arm."""

    def __init__(self, model, world, **fake):
        self.model, self.world, self.fake = model, world, fake
        self.active = None

    def start_torque_control(self):
        world = self.world
        arrays = {"data": world._address, "qpos": world.qpos.ctypes.data,
                  "qvel": world.qvel.ctypes.data, "ctrl": world.ctrl.ctypes.data,
                  "site_xpos": world.site_xpos.ctypes.data, "site_xmat": world.site_xmat.ctypes.data}
        self.active = load_native()._FakeActiveControl(
            self.model._address, arrays, _mujoco_functions()["mj_step"], **self.fake)
        return self.active

    def stop(self):
        pass

    def read_once(self):
        world = self.world
        return types.SimpleNamespace(q=list(world.qpos), dq=list(world.qvel), tau_J_d=list(world.ctrl))


def fake_robot(**fake):
    """A RobotInterface on a fake real robot; its arm is robot.robot.world."""
    robot = simulated_robot(MODEL)
    world = mujoco.MjData(MODEL)
    world.qpos[:] = HOME
    mujoco.mj_forward(MODEL, world)
    robot.real = True
    robot.robot = _StubPylibfrankaRobot(MODEL, world, **fake)
    return robot


def parameters(function):
    """What calls depend on: names, kinds and defaults (not annotations)."""
    return [(p.name, p.kind, p.default) for p in inspect.signature(function).parameters.values()]


def quietly(coroutine):
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(coroutine)


@needs_native
class ApiTest(unittest.TestCase):
    def test_has_every_attribute_and_method_of_franka_controller(self):
        python = FrankaController(simulated_robot(MODEL))
        native = NativeFrankaController(simulated_robot(MODEL))
        for name in vars(python):
            self.assertTrue(hasattr(native, name), name)
        for name, member in inspect.getmembers(FrankaController, inspect.isfunction):
            if name.startswith("__"):
                continue
            self.assertTrue(hasattr(native, name), name)
            if not name.startswith("_"):
                self.assertEqual(parameters(getattr(NativeFrankaController, name)), parameters(member), name)

    def test_attributes_read_back_as_assigned(self):
        controller = NativeFrankaController(simulated_robot(MODEL))
        controller.kp = np.arange(7.0)
        np.testing.assert_array_equal(controller.kp, np.arange(7.0))
        controller.kd = 3.0  # broadcast, as numpy would in FrankaController's law
        np.testing.assert_array_equal(controller.kd, np.full(7, 3.0))
        controller.kd[2] = 5.0  # a view of what the loop reads
        self.assertEqual(controller.kd[2], 5.0)
        target = np.eye(4)
        target[:3, 3] = [0.4, 0.1, 0.5]
        controller.ee_desired = target
        np.testing.assert_array_equal(controller.ee_desired, target)
        controller.torque_diff_limit = 500
        self.assertEqual(controller.torque_diff_limit, 500)
        controller.clip = False
        self.assertIs(controller.clip, False)
        with self.assertRaises(ValueError):
            controller.kp = np.ones(6)
        with self.assertRaises(ValueError):
            controller.switch("no such type")

    def test_refuses_a_step_override(self):
        class Collector(NativeFrankaController):
            def step(self):
                super().step()

        with self.assertRaises(TypeError):
            Collector(simulated_robot(MODEL))


@needs_native
class ParityTest(unittest.TestCase):
    """Lockstep: the same torques as FrankaController, cycle by cycle."""

    def setUp(self):
        rng = np.random.default_rng(7)
        low, high = MODEL.jnt_range[:, 0], MODEL.jnt_range[:, 1]
        self.qpos = low + (0.2 + 0.6 * rng.random(7)) * (high - low)
        self.kp, self.kd = rng.uniform(20, 120, 7), rng.uniform(2, 8, 7)
        self.offset = rng.uniform(-0.2, 0.2, 7)

    def lockstep(self, setup, steps=300):
        controllers = [cls(robot_at(self.qpos)) for cls in (FrankaController, NativeFrankaController)]
        for controller in controllers:
            setup(controller)
        python, native = controllers
        torque_error = position_error = 0.0
        for _ in range(steps):
            python.step()
            native.step()
            torque_error = max(torque_error, np.abs(python.last_command - native.last_command).max())
            position_error = max(position_error, np.abs(python.robot.data.qpos - native.robot.data.qpos).max())
        self.assertGreater(np.abs(python.last_command).max(), 0.1)  # it did push
        return torque_error, position_error

    def impedance(self, controller):
        controller.switch("impedance")
        controller.kp, controller.kd = self.kp, self.kd
        controller.q_desired = controller.q_desired + self.offset

    def osc(self, controller, tcp=None, null_offset=None, rotation=0.3):
        controller.switch("osc")
        controller.ee_kp = np.array([300, 300, 300, 600, 600, 600.0])
        controller.ee_kd = 2 * np.sqrt(controller.ee_kp)
        controller.null_kp, controller.null_kd = np.full(7, 2.0), np.full(7, 1.0)
        if tcp is not None:
            controller.set_tcp(tcp)
        if null_offset is not None:
            controller.initial_qpos = controller.initial_qpos + null_offset
        target = controller.ee_desired.copy()
        target[:3, 3] += [0.04, -0.03, 0.02]
        target = target @ pose_about_z(rotation)
        controller.ee_desired = target

    def test_impedance_is_exact(self):
        self.assertEqual(self.lockstep(self.impedance), (0.0, 0.0))

    def test_impedance_without_clip_is_exact(self):
        def setup(controller):
            self.impedance(controller)
            controller.clip = False
        self.assertEqual(self.lockstep(setup), (0.0, 0.0))

    def test_pid_is_exact(self):
        def setup(controller):
            controller.switch("pid")
            controller.kp, controller.kd, controller.ki = np.full(7, 60.0), np.full(7, 5.0), np.full(7, 20.0)
            controller.q_desired = controller.q_desired + self.offset / 2
        self.assertEqual(self.lockstep(setup), (0.0, 0.0))

    def test_osc_matches_to_rounding(self):
        tcp = np.eye(4)
        tcp[:3, 3] = [0.02, -0.01, 0.12]
        cases = {"flange": {}, "tcp": {"tcp": tcp}, "null target": {"null_offset": 0.2},
                 "half turn": {"rotation": 3.0}}
        for name, kwargs in cases.items():
            with self.subTest(name):
                torque_error, position_error = self.lockstep(lambda c: self.osc(c, **kwargs))
                self.assertLess(torque_error, 1e-9)
                self.assertLess(position_error, 1e-12)

    def test_osc_orthogonalizes_a_target_like_scipy(self):
        def setup(controller):
            self.osc(controller)
            target = controller.ee_desired.copy()
            target[:3, :3] *= 1.0 + 1e-6  # not a rotation: scipy takes the nearest one
            target[0, 1] += 1e-4
            controller.ee_desired = target
        torque_error, _ = self.lockstep(setup, steps=20)
        self.assertLess(torque_error, 1e-9)

    def test_torque_mode_starts_from_the_last_impedance_torque(self):
        def setup(controller):
            self.impedance(controller)
            for _ in range(5):
                controller.step()
            controller.switch("torque")
        python, native = (cls(robot_at(self.qpos)) for cls in (FrankaController, NativeFrankaController))
        for controller in (python, native):
            setup(controller)
            controller.step()
        np.testing.assert_array_equal(native.last_command, python.last_command)
        native.torque = np.full(7, 0.25)
        native.step()
        np.testing.assert_array_equal(native.last_command, np.full(7, 0.25))


@needs_native
class NativeSetTcpTest(test_controller.SetTcpTest):
    """test_controller's tests, the threaded one included, with the native loop."""

    controller_cls = NativeFrankaController if NATIVE_ERROR is None else None


@needs_native
class SimulationLoopTest(unittest.TestCase):
    def test_move_and_switch_while_running(self):
        robot = robot_at(HOME)
        controller = NativeFrankaController(robot)
        target = HOME + np.array([0.15, -0.1, 0.1, 0.1, 0.0, -0.1, 0.2])

        async def run():
            await controller.start()
            try:
                await controller.move(target)
                reached = np.abs(robot.data.qpos - target).max()
                held = robot.data.qpos.copy()
                for mode in ("osc", "pid", "impedance"):
                    controller.switch(mode)
                    await asyncio.sleep(0.3)
                return reached, np.abs(robot.data.qpos - held).max(), controller.state
            finally:
                await controller.stop()

        reached, drift, state = quietly(run())
        stats = controller.loop_stats()
        self.assertGreater(stats["count"], 1000)
        self.assertLess(stats["mean"], 1.5e-3)
        self.assertLess(reached, controller.arrival_tolerance)
        self.assertLess(drift, 0.01)  # switching holds the arm where it is
        self.assertEqual(sorted(state), ["ee", "jac", "last_torque", "mm", "qpos", "qvel"])

    def test_set_load_while_running_reaches_the_loop(self):
        robot = robot_at(HOME)
        controller = NativeFrankaController(robot)

        async def run():
            await controller.start()
            try:
                before = controller.state["mm"]
                robot.set_load(2.0, [0.0, 0.0, 0.1], [0.01, 0.01, 0.01])
                await asyncio.sleep(0.05)
                expected = np.zeros((7, 7))
                mujoco.mj_fullM(robot.model, robot.data, expected)
                return before, controller.state["mm"], expected
            finally:
                await controller.stop()

        before, after, expected = quietly(run())
        self.assertGreater(np.abs(after - before).max(), 1e-3)
        np.testing.assert_allclose(after, expected, atol=1e-4)  # at a pose 1 ms apart


@needs_native
class FakeRobotTest(unittest.TestCase):
    """The real-robot path: libfranka calls, robot states and errors."""

    def test_moves_and_publishes_robot_states(self):
        import pylibfranka

        robot = fake_robot()
        controller = NativeFrankaController(robot)
        target = HOME + 0.1

        async def run():
            await controller.start()
            try:
                await controller.move(target)
                state = robot.state  # from the loop: must not read a state of its own
                with self.assertRaises(RuntimeError):
                    robot.step(np.zeros(7))
                return state, robot.robot_state, robot.robot.active
            finally:
                await controller.stop()

        state, robot_state, active = quietly(run())
        world = robot.robot.world
        self.assertLess(np.abs(world.qpos - target).max(), controller.arrival_tolerance)
        self.assertEqual(active.reads, active.writes)  # every state read got a command
        np.testing.assert_allclose(state["qpos"], world.qpos, atol=1e-3)
        self.assertIsNone(robot.torque_controller)
        if load_native().has_pylibfranka_types():
            self.assertIsInstance(robot_state, pylibfranka.RobotState)
            self.assertEqual(robot_state.robot_mode, pylibfranka.RobotMode.Move)
            np.testing.assert_allclose(robot_state.q, world.qpos, atol=1e-3)
        else:
            # Built with other pybind11 internals than pylibfranka, the loop can only update
            # a RobotState that RobotInterface read, and this fake robot has none.
            self.assertIsNone(robot_state)

    def test_python_stalls_do_not_delay_the_commands(self):
        def hold_the_gil(seconds):
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                sum(i * i for i in range(1000))

        robot = fake_robot()
        controller = NativeFrankaController(robot)

        async def run():
            await controller.start()
            try:
                active = robot.robot.active
                active.reset_max_gap()
                time.sleep(0.3)  # blocks the event loop, as planning in it would
                hog = threading.Thread(target=hold_the_gil, args=(0.3,))
                hog.start()
                hog.join()
                return active.max_gap
            finally:
                await controller.stop()

        # FrankaController leaves gaps of 300 ms here, which stop the robot.
        self.assertLess(quietly(run()), 0.010)

    def test_restarts(self):
        robot = fake_robot()
        controller = NativeFrankaController(robot)

        async def run():
            for _ in range(2):
                await controller.start()
                self.assertTrue(controller.running)
                await controller.stop()
                self.assertFalse(controller.running)

        quietly(run())

    def test_an_error_stops_the_loop_and_exits_like_franka_controller(self):
        robot = fake_robot(fail_after=200, fail_message="Reflex: joint_velocity_violation (fake)")
        controller = NativeFrankaController(robot)
        errors = []
        controller.error_callback = errors.append

        async def run():
            await controller.start()
            await asyncio.sleep(2)

        with self.assertRaises(SystemExit):
            quietly(run())
        self.assertEqual(errors, ["Reflex: joint_velocity_violation (fake)"])
        self.assertFalse(controller.running)


@needs_native
@unittest.skipUnless(HAS_NUMBA, "custom control laws need numba")
class CustomLawTest(unittest.TestCase):
    def test_a_law_written_in_python_matches_the_builtin_one(self):
        @control_law
        def joint_impedance(s, p, m, tau):
            tau[:] = (p.q_desired - s.qpos) * p.kp - s.qvel * p.kd

        offset = np.linspace(-0.1, 0.1, 7)
        builtin, custom = NativeFrankaController(robot_at(HOME)), NativeFrankaController(robot_at(HOME))
        builtin.switch("impedance")
        custom.switch(joint_impedance)
        for controller in (builtin, custom):
            controller.kp, controller.kd = np.full(7, 70.0), np.full(7, 5.0)
            controller.q_desired = controller.q_desired + offset
        for _ in range(200):
            builtin.step()
            custom.step()
            np.testing.assert_allclose(custom.last_command, builtin.last_command, rtol=0, atol=1e-12)

    def test_params_and_memory(self):
        @control_law(params={"stiffness": 7}, memory={"integral": 7, "ticks": 1})
        def integrating(s, p, m, tau):
            error = p.q_desired - s.qpos
            m.integral[:] += error * s.dt
            m.ticks[0] += 1
            tau[:] = p.stiffness * error - 5.0 * s.qvel

        controller = NativeFrankaController(robot_at(HOME))
        controller.switch(integrating)
        controller.stiffness = 60.0  # the law's parameter is a controller attribute
        np.testing.assert_array_equal(controller.stiffness, np.full(7, 60.0))
        controller.q_desired = controller.q_desired + 0.05
        for _ in range(10):
            controller.step()
        memory = controller.law_memory
        self.assertEqual(memory["ticks"][0], 10)
        self.assertGreater(memory["integral"].min(), 0.0)
        controller.switch("impedance")
        controller.switch("integrating")
        controller.step()
        self.assertEqual(controller.law_memory["ticks"][0], 1)  # switch() zeroes it

        @control_law(params={"stiffness": 6})
        def clash(s, p, m, tau):
            pass

        with self.assertRaises(ValueError):
            controller.register_law(clash)

    def test_switching_back_and_forth_while_running(self):
        @control_law
        def hold(s, p, m, tau):
            tau[:] = (p.q_desired - s.qpos) * p.kp - s.qvel * p.kd

        controller = NativeFrankaController(robot_at(HOME))
        controller.register_law(hold)
        with self.assertRaises(ValueError):
            controller._loop.assign_mode(4, -1)  # a custom mode always comes with its law

        async def run():
            await controller.start()
            try:
                end = time.perf_counter() + 0.5
                switches = 0
                while time.perf_counter() < end:
                    controller.type = "hold" if switches % 2 else "impedance"
                    switches += 1
                    time.sleep(20e-6)
                await asyncio.sleep(0.01)
                return switches, controller.running
            finally:
                await controller.stop()

        switches, running = quietly(run())
        self.assertGreater(switches, 1000)
        self.assertTrue(running)

    def test_a_nonzero_return_stops_the_loop(self):
        @control_law
        def failing(s, p, m, tau):
            if s.cycle >= 50:
                return 3
            return 0

        controller = NativeFrankaController(robot_at(HOME))
        controller.switch(failing)
        errors = []
        controller.error_callback = errors.append

        async def run():
            await controller.start()
            await asyncio.sleep(1.5)

        with self.assertRaises(SystemExit):
            quietly(run())
        self.assertEqual(errors, ["control law 'failing' returned 3"])


@needs_native
class ServerControllerTest(unittest.TestCase):
    def setUp(self):
        from aiofranka.ipc import StateBlock

        self.ip = f"test.native.{os.getpid()}"
        self.shm = StateBlock(self.ip, create=True)

    def tearDown(self):
        self.shm.close()
        self.shm.unlink()

    def test_writes_the_state_to_shared_memory(self):
        from aiofranka.ipc import STATUS_RUNNING
        from aiofranka.server_native import NativeServerController

        robot = robot_at(HOME)
        controller = NativeServerController(robot, self.shm)

        async def run():
            task = await controller.start()
            first = self.shm.read_state()
            await asyncio.sleep(0.2)
            second = self.shm.read_state()
            controller.running = False  # what the server's stop command does
            await asyncio.wait_for(task, 2.0)
            return first, second

        first, second = quietly(run())
        self.assertEqual(self.shm.read_status(), STATUS_RUNNING)
        self.assertGreater(second["timestamp"], first["timestamp"])
        np.testing.assert_allclose(second["qpos"], robot.data.qpos, atol=1e-3)
        np.testing.assert_allclose(second["q_desired"], controller.q_desired)

    def test_reports_an_error_without_exiting(self):
        from aiofranka.ipc import STATUS_ERROR
        from aiofranka.server_native import NativeServerController

        robot = fake_robot(fail_after=100, fail_message="Reflex: fake")
        controller = NativeServerController(robot, self.shm)

        async def run():
            task = await controller.start()
            await asyncio.wait_for(task, 3.0)

        quietly(run())
        self.assertEqual(controller._last_error, "Reflex: fake")
        self.assertEqual(self.shm.read_status(), STATUS_ERROR)
        self.assertEqual(self.shm.read_error(), "Reflex: fake")


if __name__ == "__main__":
    unittest.main()
