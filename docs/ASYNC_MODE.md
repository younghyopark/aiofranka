# Async Mode Guide

> **Using `Controller` (the default)?** You can skip this entire document. Its calls are plain, and its event loop runs in a thread of its own, where your code can't block it.

This guide is for asyncio code with `NativeFrankaController`. There, `controller.start()` starts the 1 kHz loop, and your script runs on the asyncio event loop next to it. `NativeFrankaController`'s loop runs in C++ and keeps sending torques whatever the event loop does, so blocking the event loop does not stop the robot. It delays what runs there: your next `set()`, a `move()` (which streams its trajectory from the event loop), and the copy of the loop's state into `robot.data` and `robot.robot_state`. This guide keeps the event loop responsive.

With the legacy `FrankaController`, whose 1 kHz loop itself runs on the event loop, the rules below are required.

## The Core Rule

After `controller.start()`, **don't block the asyncio event loop**:

- With `NativeFrankaController`, the arm holds its last target until the event loop runs again. `controller.state` reads the loop directly and stays current.
- With the legacy `FrankaController`, the 1 kHz loop starves: the robot triggers a `communication_constraints_violation` reflex and aborts the motion.

## What Blocks the Event Loop

Any synchronous (non-awaiting) work blocks it. With the native loop, it delays your targets by as long; more than ~1ms starves the legacy loop:

```python
# BAD — blocks the event loop
await controller.start()
result = model(input_tensor)          # 2ms+ of GPU compute
frame = cv2.imread("image.png")       # disk I/O
time.sleep(0.1)                       # synchronous sleep
data = requests.get("http://...")     # network I/O
```

## How to Fix It

aiofranka provides `asyncify` to offload blocking work to a thread executor:

```python
from aiofranka import asyncify

# As a decorator — turns any function/method into an awaitable
class MyPolicy:
    @asyncify
    def get_action(self, obs):
        return self.model(obs)  # blocking, but now safe to await

# Wrapping an existing function you don't own
model_async = asyncify(model)

# Now safe to use in the control loop
await controller.start()
for step in range(100):
    action = await policy.get_action(obs)          # non-blocking
    result = await model_async(input_tensor)       # non-blocking
    await controller.set("ee_desired", action)     # non-blocking
```

The original sync function is still accessible as `policy.get_action.sync(obs)` if needed.

## `time.sleep` vs `asyncio.sleep`

A common gotcha: `time.sleep()` blocks the entire event loop, which holds your targets, and **will** starve the legacy 1kHz control loop. Always use `await asyncio.sleep()` instead:

```python
# BAD — blocks the event loop
time.sleep(0.1)

# GOOD — yields control back to the event loop
await asyncio.sleep(0.1)
```

This applies anywhere after `controller.start()`. Even a short `time.sleep(0.002)` can cause a violation.

## Quick Checklist

| Safe (non-blocking)                     | Unsafe (blocking)                        |
|-----------------------------------------|------------------------------------------|
| `await asyncio.sleep(dt)`               | `time.sleep(dt)`                         |
| `await loop.run_in_executor(None, fn)`  | `fn()` directly if `fn` takes >1ms      |
| `await controller.set(...)`             | Heavy computation inline                 |
| `await controller.move(...)`            | `cv2.imread(...)`, `np.load(...)` on large files |

## CUDA and `run_in_executor`

If your model runs on GPU, **avoid `run_in_executor`** for CUDA operations. The default `ThreadPoolExecutor` causes GIL contention between the worker thread (launching CUDA kernels) and the main thread (running the event loop). This can inflate a 1ms forward pass to 50ms+.

```python
# BAD — GIL contention makes CUDA ~40x slower in a worker thread
ee_desired = await loop.run_in_executor(None, model, input_tensor)

# GOOD — if your GPU forward pass is <2ms, call it directly
ee_desired = model(input_tensor)  # fast enough not to hold up the event loop
```

**Rule of thumb**: profile your model's forward pass with the warmup loop. If it's under ~2ms on GPU, call it directly on the event loop. If it's slower (e.g., large vision models), consider running on CPU or using a separate process instead of a thread executor.

## Initialization Order

Do heavy setup (model loading, CUDA warmup, calibration) **before** `controller.start()`:

```python
# 1. Load model, warm up CUDA, connect cameras — all before start()
model = load_model(checkpoint)
warm_up_inference(model)

# 2. Now start the real-time loop
await controller.start()
await controller.move()

# 3. From here on, only use await-based calls
```
