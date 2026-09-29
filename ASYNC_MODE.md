# Async Mode Guide

> **Using server mode (`FrankaRemoteController`)?** You can skip this entire document. The 1kHz loop runs in a separate process, so your script can't starve it.

This guide covers the sharp edges of **async mode** (`FrankaController`) — the in-process control loop that requires careful async discipline.

## The Core Rule

After `controller.start()`, a **1kHz async control loop** runs in the background communicating with the robot via libfranka. If this loop is starved (i.e., doesn't get to run on time), the robot triggers a `communication_constraints_violation` reflex and aborts the motion.

**Never block the asyncio event loop after `controller.start()`.**

## What Blocks the Event Loop

Any synchronous (non-awaiting) work that takes more than ~1ms will starve the 1kHz loop:

```python
# BAD — blocks the event loop, will trigger reflex
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

A common gotcha: `time.sleep()` blocks the entire event loop, which **will** starve the 1kHz control loop. Always use `await asyncio.sleep()` instead:

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

If your model runs on GPU, **avoid `run_in_executor`** for CUDA operations. The default `ThreadPoolExecutor` causes GIL contention between the worker thread (launching CUDA kernels) and the main thread (running the 1kHz loop). This can inflate a 1ms forward pass to 50ms+.

```python
# BAD — GIL contention makes CUDA ~40x slower in a worker thread
ee_desired = await loop.run_in_executor(None, model, input_tensor)

# GOOD — if your GPU forward pass is <2ms, call it directly
ee_desired = model(input_tensor)  # fast enough to not starve the 1kHz loop
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
