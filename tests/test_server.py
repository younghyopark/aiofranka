"""
The control loop the server and the CLI run: the native one by default, the legacy Python
one when asked, or where the native one is not built.
"""

import unittest
from unittest import mock

from aiofranka.controller import FrankaController
from aiofranka.remote import FrankaRemoteController
from aiofranka.server import ServerController, _controller_class, _server_controller_class

try:
    from aiofranka.native import NativeFrankaController, load_native

    load_native()
    NATIVE_ERROR = None
except ImportError as error:
    NATIVE_ERROR = str(error)


class DefaultLoopTest(unittest.TestCase):
    @unittest.skipIf(NATIVE_ERROR is not None, f"native loop not built: {NATIVE_ERROR}")
    def test_runs_the_native_loop_by_default(self):
        from aiofranka.server_native import NativeServerController

        self.assertIs(_server_controller_class(None), NativeServerController)
        self.assertIs(_server_controller_class("native"), NativeServerController)
        self.assertIs(_controller_class(), NativeFrankaController)

    def test_runs_the_python_loop_when_asked(self):
        self.assertIs(_server_controller_class("python"), ServerController)
        self.assertIs(_server_controller_class(ServerController), ServerController)

    def test_falls_back_to_the_python_loop_without_the_native_one(self):
        with mock.patch("aiofranka.native.load_native", side_effect=ImportError("not built\nhelp")):
            with self.assertLogs("aiofranka.server", "WARNING") as logs:
                self.assertIs(_server_controller_class(None), ServerController)
            self.assertIn("legacy Python loop: not built", logs.output[0])
            with self.assertLogs("aiofranka.server", "WARNING"):
                self.assertIs(_controller_class(), FrankaController)

    def test_the_remote_controller_asks_for_the_python_loop_with_native_false(self):
        self.assertIsNone(FrankaRemoteController("172.16.0.2")._server_controller)
        self.assertEqual(FrankaRemoteController("172.16.0.2", native=False)._server_controller, "python")


if __name__ == "__main__":
    unittest.main()
