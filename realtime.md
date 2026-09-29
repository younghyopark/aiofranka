# Real-Time Tuning Guide for 1 kHz Franka Torque Control

This guide documents the system-level optimizations applied to achieve reliable 1 kHz torque control on a Franka Emika Panda using `aiofranka`. These changes reduced worst-case loop jitter from **2903 μs → 1188 μs** and eliminated all iterations above 1.5 ms.

## Hardware & Software

| Component | Details |
|-----------|---------|
| Robot | Franka Emika Panda (FCI enabled) |
| Workstation | 32-core Intel (16C/32T), RTX GPU |
| OS | Ubuntu 22.04 |
| Kernel | `5.15.0-1095-realtime` (PREEMPT_RT) |
| Control stack | `aiofranka` (asyncio-based, Python) |
| RT core | CPU 31 (last logical core) |
| Franka NIC | `eno1`, direct gigabit link to `172.16.0.2` |

> **Prerequisite:** You must be running a `PREEMPT_RT` kernel. On Ubuntu, install one with:
> ```bash
> sudo apt install linux-image-5.15.0-1095-realtime linux-headers-5.15.0-1095-realtime
> ```
> Reboot and confirm with `uname -r`. If your kernel does not have `realtime` or `rt` in the name, none of the tuning below will help much — the vanilla kernel cannot guarantee sub-millisecond scheduling.

---

## Results Summary

All benchmarks: `aiofranka rt-benchmark --cpu-pin 31 --sched-fifo --duration 10`

| Metric | Before tuning | After tuning | Change |
|--------|--------------|--------------|--------|
| std | 55.9 μs | 34.4 μs | 1.6× tighter |
| p99 | 1089 μs | 1085 μs | — |
| p99.9 | 1622 μs | 1104 μs | now in spec |
| max | 2903 μs | 1188 μs | 2.4× better |
| jitter (max−min) | 2823 μs | 292 μs | 9.7× better |
| out-of-spec (>1100 μs) | 76 / 9999 | 31 / 9999 | 2.5× fewer |
| iterations > 1500 μs | 11 | 0 | eliminated |
| iterations > 2000 μs | 4 | 0 | eliminated |

---

## Step 1 — GRUB (kernel boot parameters)

These take effect at boot and cannot be changed at runtime. Edit `/etc/default/grub`:

```bash
sudo nano /etc/default/grub
```

Set the `GRUB_CMDLINE_LINUX` line to:

```
GRUB_CMDLINE_LINUX="isolcpus=31 nohz_full=31 rcu_nocbs=31 processor.max_cstate=1 intel_idle.max_cstate=0 pcie_aspm=off"
```

> **Adapt `31` to your setup.** Pick a core that is not core 0, ideally the last core on the last NUMA node. Use `lscpu` to check topology. If your machine has fewer than 32 threads, adjust accordingly.

Then apply and reboot:

```bash
sudo update-grub
sudo reboot
```

Verify after reboot:

```bash
cat /proc/cmdline
# Should contain all six parameters
```

### What each parameter does

**`isolcpus=31`** — Removes the core from the kernel's general-purpose scheduler. Without this, any userspace process (desktop services, background daemons, etc.) can be scheduled onto the RT core and preempt the control loop. With `isolcpus`, only processes explicitly pinned to the core (via `taskset` or `cpu_affinity`) will run there.

**`nohz_full=31`** — Disables the periodic scheduler tick on the core when only one task is running. The default kernel tick (typically 250 Hz or 1 kHz) generates a timer interrupt that costs a few microseconds of jitter each time. Since the control loop is the sole occupant of the isolated core, the tick is unnecessary.

**`rcu_nocbs=31`** — Offloads RCU (Read-Copy-Update) callback processing to other cores. RCU is a kernel synchronization mechanism that periodically runs cleanup work on every core. These callbacks are unpredictable in timing and can take tens of microseconds. This flag ensures they never interrupt the RT core.

**`processor.max_cstate=1` / `intel_idle.max_cstate=0`** — Prevents the CPU from entering deep sleep states. When the RT core is idle (e.g., waiting for the next Franka state packet in `readOnce`), the CPU would normally enter a deep C-state to save power. Exiting deep C-states (C3, C6) can take 50–200+ μs. Capping at C-state 1 keeps wakeup latency near zero. Both parameters are needed to cover different kernel idle drivers.

**`pcie_aspm=off`** — Disables PCIe Active State Power Management. ASPM allows PCIe devices (including the NIC) to enter low-power link states. When a packet arrives on a sleeping link, there is a wakeup penalty of tens of microseconds. Disabling ASPM keeps the NIC's PCIe link fully active.

---

## Step 2 — Runtime tuning (applied after every boot)

These settings do not persist across reboots. Apply them manually or via a systemd service (see Step 3).

```bash
# Set CPU governor to performance (prevent frequency scaling)
echo performance | sudo tee /sys/devices/system/cpu/cpu31/cpufreq/scaling_governor

# Disable RT throttling
# Default: kernel reserves 5% of each second (50 ms) for non-RT tasks,
# forcibly descheduling your SCHED_FIFO thread. This causes periodic multi-ms stalls.
echo -1 | sudo tee /proc/sys/kernel/sched_rt_runtime_us

# Disable NMI watchdog (eliminates periodic performance-monitoring interrupts)
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog

# Disable NIC interrupt coalescing on the Franka interface
# Default may batch interrupts for up to 256 μs — unacceptable for 1 kHz control.
sudo ethtool -C eno1 rx-usecs 0 rx-frames 1

# Stop irqbalance (it overrides manual IRQ affinity settings)
sudo systemctl stop irqbalance
sudo systemctl disable irqbalance
```

### What each setting does

**CPU governor → `performance`** — Locks the core at maximum frequency. The default `powersave` governor dynamically scales frequency, and transitions between P-states take microseconds. For a 1 kHz loop with ~70 μs of compute per iteration, even small frequency transitions are significant.

**RT throttling → disabled** — By default, `sched_rt_runtime_us=950000` and `sched_rt_period_us=1000000` means the kernel gives RT tasks only 950 ms out of every 1000 ms, forcibly preempting them for 50 ms to prevent RT tasks from starving the system. On an isolated core with a single pinned task, this protection is unnecessary and causes periodic multi-millisecond gaps.

**NMI watchdog → off** — The NMI watchdog fires a non-maskable interrupt via the CPU's performance monitoring unit to detect lockups. This interrupt cannot be avoided by any scheduling priority and adds jitter. On a dedicated RT core, the watchdog serves no purpose.

**NIC interrupt coalescing → immediate** — By default, many NIC drivers batch incoming packet notifications to improve throughput. Setting `rx-usecs=0` and `rx-frames=1` tells the NIC to fire an interrupt immediately when a packet arrives, minimizing the delay between the Franka sending a state update and the control loop waking up to process it.

**irqbalance → stopped** — The `irqbalance` daemon periodically redistributes hardware interrupt affinities across cores for load balancing. This undoes any manual IRQ pinning. Disabling it ensures your NIC IRQ affinity settings stay put.

---

## Step 3 — Persist runtime settings with systemd

Create a oneshot service so the runtime tweaks survive reboots:

```bash
sudo tee /etc/systemd/system/rt-tuning.service << 'EOF'
[Unit]
Description=Real-time latency tuning for Franka 1kHz control
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  echo performance > /sys/devices/system/cpu/cpu31/cpufreq/scaling_governor; \
  echo -1 > /proc/sys/kernel/sched_rt_runtime_us; \
  echo 0 > /proc/sys/kernel/nmi_watchdog; \
  ethtool -C eno1 rx-usecs 0 rx-frames 1; \
  systemctl stop irqbalance 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rt-tuning.service
```

Verify after next reboot:

```bash
sudo systemctl status rt-tuning.service
cat /sys/devices/system/cpu/cpu31/cpufreq/scaling_governor   # → performance
cat /proc/sys/kernel/sched_rt_runtime_us                      # → -1
cat /proc/sys/kernel/nmi_watchdog                             # → 0
ethtool -c eno1 | grep rx-usecs                               # → 0
```

---

## Diagnostics

### Audit script

Run `sudo bash rt_audit.sh` (included in this repo) to check all settings at a glance. It covers kernel config, CPU isolation, frequency scaling, NIC coalescing, IRQ affinity, memory settings, power management, and runs a quick `cyclictest` baseline.

### Benchmarking

```bash
aiofranka rt-benchmark --cpu-pin 31 --sched-fifo --duration 10
```

Key things to look at:
- **p99.9** should be under 1100 μs
- **max** should be under 1200 μs (ignoring iteration 0 which is a cold-start outlier)
- **>1500 μs iterations** should be zero
- **Jitter attribution** should show no single phase dominating with large spikes

### cyclictest (kernel-level baseline)

```bash
sudo cyclictest -t1 -p80 -a31 -i1000 -D10 -q
```

On a properly tuned system, max wakeup latency should be under 10 μs. If `cyclictest` shows good numbers but `aiofranka` doesn't, the bottleneck is in userspace (asyncio event loop overhead).

Install with: `sudo apt install rt-tests`

---

## Troubleshooting

**Spikes > 2 ms that correlate with `readOnce`** — Almost always a system-level issue: check `isolcpus`, NIC coalescing, and `irqbalance`. These were responsible for the original 2.9 ms spikes.

**Periodic stalls every ~1 second** — RT throttling (`sched_rt_runtime_us`). Check that it's set to `-1`.

**Jitter improves then regresses after reboot** — Runtime settings didn't persist. Check that `rt-tuning.service` is enabled and running.

**Governor resets to `powersave`** — Some systems have `thermald` or `power-profiles-daemon` overriding the governor. Check with `systemctl status thermald power-profiles-daemon` and disable if needed.

**IRQ threads still on the RT core despite `isolcpus`** — Kernel IRQ threads can bypass `isolcpus`. Check with `ps -eo pid,comm,psr | grep <core>`. For critical cases, manually set affinity: `echo 0 > /proc/irq/<N>/smp_affinity_list`.

**NIC coalescing resets after reboot** — The NIC driver loads its defaults at boot. This is handled by the systemd service, but if you're not using it, add the `ethtool` command to `/etc/rc.local` or a udev rule.

---

## Optional further tuning

These were not applied in our setup but may help in tighter scenarios:

- **Disable Transparent Huge Pages**: `echo never > /sys/kernel/mm/transparent_hugepage/enabled` — THP compaction can cause sporadic multi-ms stalls.
- **Pin NIC IRQ to RT core or adjacent core**: `echo 31 > /proc/irq/<N>/smp_affinity_list` — Reduces cache misses between interrupt handling and the control thread.
- **Disable turbo boost**: `echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo` — Eliminates frequency transition jitter at the cost of peak single-core performance.
- **Lock `/dev/cpu_dma_latency`**: Hold a file descriptor open with value `0` written to it — Prevents the CPU from entering any idle state deeper than C0, even if the cmdline C-state settings are insufficient.
- **Replace asyncio hot path**: If the remaining ~30 μs of std is too much, the asyncio event loop's wakeup latency is the likely floor. A raw `epoll_wait()` on the UDP socket in C/Rust would shave this further.