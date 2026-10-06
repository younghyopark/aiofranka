NativeFrankaController
======================

``NativeFrankaController`` has ``FrankaController``'s constructor, methods and attributes
(``switch()``, ``set()``, ``move()``, ``activate()``, ``identify_payload()``, ...), documented in
:doc:`controller`. Below is what it adds: its loop in C++, ``record()`` and control laws.

.. automodule:: aiofranka.native
   :members: NativeFrankaController, Recording, control_law, ControlLaw, load_native
   :show-inheritance:

.. automodule:: aiofranka.server_native
   :members: NativeServerController
   :show-inheritance:
