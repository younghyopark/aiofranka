"""
Build aiofranka's native control loop, the optional extension aiofranka._native.

The package metadata is in pyproject.toml. aiofranka works without the extension: if it
does not build (no C++ compiler, or no pybind11), only NativeFrankaController is missing.
AIOFRANKA_NATIVE=0 skips it, e.g. to build a pure-Python wheel.
"""

import os
import sys
import sysconfig

from setuptools import setup

if sys.platform == "darwin" and "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
    # pybind11 would target macOS 10.14, which current libc++ headers warn about.
    python_target = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "12.0"
    version = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = max(python_target, "12.0", key=version)

ext_modules = []
cmdclass = {}
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("pybind11 is not installed: building aiofranka without its native control loop", file=sys.stderr)
    Pybind11Extension = None
if Pybind11Extension is not None and os.environ.get("AIOFRANKA_NATIVE") != "0":
    if sys.platform == "darwin":
        # libfranka is not linked: aiofranka.native loads pylibfranka's copy first. Bind its
        # symbols at import, so a missing one fails the import instead of the control loop.
        compile_args, link_args = [], ["-Wl,-bind_at_load"]
    else:
        compile_args, link_args = ["-pthread"], ["-pthread", "-Wl,-z,now"]
    ext_modules.append(Pybind11Extension(
        "aiofranka._native",
        ["src/native/module.cpp"],
        include_dirs=["src/native", "src/native/include"],
        cxx_std=17,
        # No fused multiply-adds, which numpy does not use either: the native control laws
        # then round like FrankaController's.
        extra_compile_args=["-O2", "-ffp-contract=off", *compile_args],
        extra_link_args=link_args,
        optional=True,
    ))
    cmdclass["build_ext"] = build_ext

setup(ext_modules=ext_modules, cmdclass=cmdclass)
