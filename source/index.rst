.. aiofranka documentation master file

Welcome to aiofranka's documentation!
======================================

.. image:: ../../assets/image.png
   :width: 340
   :align: center
   :alt: aiofranka logo

.. image:: https://img.shields.io/pypi/v/aiofranka
   :alt: PyPI version
   :target: https://pypi.org/project/aiofranka/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :alt: License: MIT
   :target: https://opensource.org/licenses/MIT

**aiofranka** is an asyncio-based Python library for controlling Franka Emika robots. It provides a high-level interface that combines **pylibfranka** for real-time 1kHz torque control, **MuJoCo** for kinematics/dynamics computation, and **Ruckig** for smooth trajectory generation.

The library is designed for research applications requiring precise, real-time control with minimal latency and maximum flexibility.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   controllers
   cli
   async_mode
   examples
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/remote
   api/controller
   api/robot
   api/server
   api/gripper
   api/utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
