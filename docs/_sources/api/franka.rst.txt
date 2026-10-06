Robot and Controller
====================

``Robot`` is the arm, and ``Controller`` drives it with the native 1 kHz loop. The controller's
modes, gains, targets, control laws and recordings are those of ``NativeFrankaController``
(see :doc:`native` and :doc:`controller`), with plain calls instead of awaitable ones.

.. automodule:: aiofranka.franka
   :members: Robot, Controller
