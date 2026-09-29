# Real-Time Tuning Guide for aiofranka

This document describes the optimizations applied to the aiofranka 1kHz torque control loop to minimize jitter on PREEMPT_RT Linux systems, along with benchmark results.

## System Setup

- **CPU**: Intel Core i9-14900K (32 threads: 8 P-cores + 16 E-cores)
- **Kernel**: 5.15.0-1095-realtime (PREEMPT_RT)
- **NIC**: Aquantia/Marvell (atlantic driver) — `eno1`
- **Robot**: Franka Emika, connected via direct Ethernet at 1Gbps
- **Python**: 3.x with pylibfranka, MuJoCo, NumPy

## Control Loop Architecture

The 1kHz control loop runs as follows each iteration (~1ms):

```
readOnce()     ~927us   Blocking wait for next robot state (libfranka UDP)
mj_forward()    ~30us   Sync MuJoCo model with robot state
state_build     ~13us   Compute ee pose, Jacobian, mass matrix
ctrl_law        ~25us   Compute torques (impedance/PID/OSC)
shm_write        ~4us   Write state to shared memory for client
─────────────────────
TOTAL           ~999us
```

`readOnce()` dominates — it blocks until the Franka controller sends its next 1kHz state packet over UDP. This is the timing reference for the loop.

## Optimizations Applied

### 1. CPU Pinning (Python-side)

Pin the RT control thread to a dedicated CPU core to avoid cache thrashing from migration.

```python
os.sched_setaffinity(0, {31})  # last core
```

**Impact**: Improved p99 and reduced std by keeping the thread's cache hot.

### 2. SCHED_FIFO Real-Time Priority (Python-side)

Elevate the control thread to the highest-priority Linux scheduling class.

```python
os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
```

**Impact**: Prevents normal-priority processes from preempting the control loop. Requires `CAP_SYS_NICE` or root.

### 3. Pre-allocated NumPy Buffers (Python-side)

Reuse pre-allocated arrays instead of creating new ones every iteration. The original code allocated ~10 NumPy arrays per iteration (via `np.eye()`, `np.zeros()`, `np.array()`), creating GC pressure.

Before (allocates every iteration):
```python
ee = np.eye(4)
jac = np.zeros((6, 7))
mm = np.zeros((7, 7))
state = {"qpos": np.array(data.qpos), ...}
```

After (reuses buffers):
```python
# Allocated once before the loop
_ee = np.eye(4)
_jac = np.zeros((6, 7))
_mm = np.zeros((7, 7))

# Inside loop — write into existing arrays
_ee[:3, :3] = ee_mat
_jac[:] = 0
mujoco.mj_jacSite(model, data, _jac[:3], _jac[3:], site_id)
np.copyto(_qpos, data.qpos)
```

**Impact**: Biggest single Python-side improvement. Reduced std from 44us to 29us and max from 2318us to 1112us by eliminating per-iteration allocation and GC pauses.

### 4. NIC Interrupt Coalescing (System-level)

The Aquantia NIC was batching interrupts for up to **256us** before notifying the CPU — adding random delay to every incoming packet from the robot.

```bash
sudo ethtool -C eno1 rx-usecs 1 tx-usecs 1
```

**Impact**: Reduced worst-case network latency. This setting does not persist across reboots — see [Persistence](#making-changes-persistent) below.

### 5. NIC IRQ Pinning (System-level)

Pin all NIC hardware interrupts to a single non-RT core so they don't compete with the control thread.

```bash
# Pin all eno1 IRQs to core 0
for irq in $(grep eno1 /proc/interrupts | awk '{print $1}' | tr -d ':'); do
  echo 0 | sudo tee /proc/irq/$irq/smp_affinity_list
done
```

**Impact**: Prevents NIC interrupt handling from preempting the RT thread on core 31. Also improves cache efficiency for interrupt handling by keeping it on one core.

## Benchmark Results

All benchmarks run with `aiofranka rt-benchmark --duration 10` on the real robot.

### Python-side Optimizations (--all-combos)

| Config | In-spec (900-1100us) | std | p99 | max |
|--------|---------------------|-----|-----|-----|
| baseline | 97.76% | 49.3us | 1103us | 3718us |
| cpu=31+FIFO | 99.51% | 44.1us | 1076us | 2318us |
| cpu=31+FIFO+nogc | 99.61% | 45.0us | 1075us | 2434us |
| cpu=31+FIFO+mlock | 99.83% | 40.9us | 1077us | 2981us |
| **cpu=31+FIFO+prealloc** | **99.89%** | **29.2us** | **1073us** | **1112us** |
| cpu=31+FIFO+all | 99.68% | 46.7us | 1076us | 3677us |

Notes:
- `nogc`: `gc.disable()` during the loop — marginal improvement, GC pauses are rare.
- `mlock`: `mlockall(MCL_CURRENT | MCL_FUTURE)` — helped in-spec % but didn't help max. `MCL_FUTURE` can cause overhead from aggressive page locking.
- `prealloc`: Clear winner — eliminates per-iteration numpy allocations.
- `all` (gc+mlock+prealloc combined): Worse than prealloc alone due to `mlockall` overhead.

### System-level Optimizations

| Config | std | p99.9 | max | >2000us |
|--------|-----|-------|-----|---------|
| baseline | 49.3us | — | 3718us | 3 |
| + cpu pin + FIFO | 44.0us | 1338us | 2217us | 3 |
| + IRQ pin + rx-usecs=1 | **33.5us** | **1112us** | **1716us** | **0** |

### Jitter Attribution (Final Configuration)

For out-of-spec iterations, which phase caused the spike:

| Phase | % Blamed | Avg Excess | Max Excess |
|-------|----------|-----------|-----------|
| **readOnce** | **55.4%** | **126.7us** | **665.6us** |
| mj_fwd | 14.3% | 1.1us | 4.0us |
| state_build | 8.9% | 21.3us | 105.1us |
| ctrl_law | 10.7% | 0.7us | 1.8us |
| shm_write | 10.7% | 0.0us | 0.1us |

**Conclusion**: All remaining jitter is in `readOnce()` — the libfranka UDP blocking read that waits for the robot controller's 1kHz state packet over Ethernet. This is network/hardware jitter, not Python overhead.

## What Didn't Help

- **`gc.disable()`**: Python's GC rarely triggers during tight loops since pre-allocated buffers produce little garbage. Marginal improvement.
- **`mlockall()`**: Prevents page faults, but `MCL_FUTURE` adds overhead to every new memory mapping. Actually made max jitter worse when combined with other optimizations.
- **asyncio vs RT thread (v1 vs v2)**: The v2 dedicated RT thread showed nearly identical results to v1 asyncio (98.3% vs 97.7% in initial testing). This is because `readOnce()` is the blocking synchronization point — asyncio's `await sleep(0)` overhead is negligible compared to the 927us blocking read.

## Remaining Optimization Opportunities

These were not applied but would provide additional improvements:

### Kernel Boot Parameters (requires reboot)

```bash
# Add to /etc/default/grub GRUB_CMDLINE_LINUX:
isolcpus=31 nohz_full=31 rcu_nocbs=31
```

- `isolcpus=31`: Prevents the scheduler from placing any other task on core 31.
- `nohz_full=31`: Disables periodic timer ticks on core 31 (adaptive-ticks mode).
- `rcu_nocbs=31`: Offloads RCU callbacks from core 31 to other cores.

After editing grub:
```bash
sudo update-grub
sudo reboot
```

### Network Stack Tuning

```bash
# Increase socket buffer size
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.rmem_default=16777216

# Disable GRO/GSO (reduces batching latency)
sudo ethtool -K eno1 gro off gso off
```

## Making Changes Persistent

The runtime changes (ethtool, IRQ pinning) do not survive reboots. To persist them:

### Option A: systemd service

Create `/etc/systemd/system/aiofranka-rt-tuning.service`:

```ini
[Unit]
Description=RT tuning for aiofranka
After=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  ethtool -C eno1 rx-usecs 1 tx-usecs 1; \
  for irq in $(grep eno1 /proc/interrupts | awk "{print \\$1}" | tr -d ":"); do \
    echo 0 > /proc/irq/$irq/smp_affinity_list; \
  done'

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl enable aiofranka-rt-tuning
```

### Option B: rc.local or udev rules

Add the commands to `/etc/rc.local` or create a udev rule that triggers when `eno1` comes up.

## Benchmarking

Use the built-in benchmark tool to measure your system's RT performance:

```bash
# Basic benchmark
aiofranka rt-benchmark --duration 10

# With RT tuning
aiofranka rt-benchmark --cpu-pin 31 --sched-fifo --duration 10

# Compare all combinations automatically
aiofranka rt-benchmark --all-combos --duration 10
```

The benchmark runs a 1kHz impedance hold loop on the real robot and reports:
- Iteration timing statistics (mean, std, percentiles, max)
- Per-phase breakdown (readOnce, mj_fwd, state_build, ctrl_law, shm_write)
- Jitter attribution (which phase caused each out-of-spec iteration)
- Top 10 worst iterations with full phase detail
- ASCII histogram of iteration time distribution
