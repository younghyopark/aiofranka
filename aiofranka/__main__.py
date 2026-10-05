"""
CLI entry point for aiofranka.

Usage:
    aiofranka start-server [--ip IP] [--foreground] [--no-home]
    aiofranka stop [--ip IP]
    aiofranka status [--ip IP]
    aiofranka mode [--set MODE]
    aiofranka log [-n LINES] [--follow]
    aiofranka camera calibrate|fit
"""

import argparse
import getpass
import json
import os
import signal
import sys
import time

DEFAULT_IP = "172.16.0.2"
CONFIG_DIR = os.path.expanduser("~/.aiofranka")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

# ── Pretty output ──────────────────────────────────────────────────────────

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str) -> str:
    """Return ANSI code if stdout is a terminal, else empty string."""
    return code if _IS_TTY else ""


BOLD = _c("\033[1m")
DIM = _c("\033[2m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
RED = _c("\033[31m")
RST = _c("\033[0m")


def _step_line(step: int, total: int, label: str, status: str) -> str:
    tag = f"{DIM}[{step}/{total}]{RST}"
    ndots = max(2, 38 - len(label))
    dots = " " + "." * ndots
    return f"  {tag} {label}{dots} {status}"


def _wrap(text: str, width: int = 60, indent: str = "  ") -> str:
    """Word-wrap text to fit in terminal, with indent on each line."""
    import textwrap
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent,
                      subsequent_indent=indent)
        for line in text.splitlines()
    )


def _get_version() -> str:
    try:
        from aiofranka import __version__
        return __version__
    except Exception:
        return "?"


# ── Config helpers ─────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(config: dict):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _resolve_ip(args_ip: str | None) -> str:
    """Resolve IP: CLI arg > config file > default."""
    if args_ip:
        config = _load_config()
        config["last_ip"] = args_ip
        _save_config(config)
        return args_ip
    config = _load_config()
    return config.get("last_ip", DEFAULT_IP)


def _resolve_credentials(args) -> tuple[str, str]:
    """Resolve credentials: CLI flags > saved config > interactive prompt."""
    config = _load_config()
    username = args.username
    password = args.password

    if username == "admin" and password == "admin":
        saved_user = config.get("username")
        saved_pass = config.get("password")
        if saved_user and saved_pass:
            return saved_user, saved_pass

        print(f"  Robot web UI credentials {DIM}(saved to ~/.aiofranka/config.json){RST}")
        username = input("  Username [admin]: ").strip() or "admin"
        password = getpass.getpass("  Password: ")
        print()

    config["username"] = username
    config["password"] = password
    _save_config(config)
    return username, password


def _check_control_token(robot_ip: str, username: str, password: str,
                         protocol: str) -> bool:
    """Pre-check control token before daemonizing.

    Returns True if user chose to proceed without the control token,
    False if token is available (or robot isn't pre-unlocked).
    """
    from aiofranka.server import _DeskClientV2
    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
        if not (client.are_joints_unlocked() and client.is_fci_active()):
            return False  # robot needs full unlock, daemon will handle it
        # Robot is already unlocked + FCI active — try to get token
        client.take_token(timeout=5)
        client.release_token(best_effort=True)  # release so the daemon can take it
        return False
    except Exception:
        # Token held by someone else — check if the old server is dead
        from aiofranka.ipc import pid_file_for_ip
        pid_path = pid_file_for_ip(robot_ip)
        old_server_dead = False
        if os.path.exists(pid_path):
            try:
                with open(pid_path) as f:
                    old_pid = int(f.read().strip())
                os.kill(old_pid, 0)
            except (ProcessLookupError, ValueError):
                old_server_dead = True
        else:
            old_server_dead = True

        if old_server_dead:
            # Stale token from crashed server — daemon will handle it
            # via timeout-based contention. Just clean up PID file.
            print(f"  {YELLOW}Stale control token detected (old server is dead).{RST}")
            print(f"  The daemon will attempt to recover it automatically.\n")
            try:
                os.unlink(pid_path)
            except FileNotFoundError:
                pass
            return False

        # Old server is still running — prompt user
        while True:
            print(f"  {YELLOW}Control token is held by someone via the web GUI.{RST}")
            print(f"  The robot is already unlocked with FCI active.\n")
            print(f"  {BOLD}[1]{RST} Release control on the web GUI first, then retry")
            print(f"  {BOLD}[2]{RST} Proceed without control token\n")
            choice = input(f"  Choose [1/2]: ").strip()
            if choice == "2":
                print()
                return True
            if choice == "1":
                input(f"\n  Release control on the web GUI, then press {BOLD}Enter{RST} to retry...")
                try:
                    client2 = _DeskClientV2(robot_ip, username, password, protocol=protocol)
                    client2.take_token(timeout=5)
                    client2.release_token(best_effort=True)
                    print(f"  {GREEN}Control token is now available.{RST}\n")
                    return False
                except Exception:
                    print(f"\n  {RED}Still can't acquire control token.{RST}\n")
                    continue
            print()
    except Exception:
        return False  # login failed or other error, let daemon handle it


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_start(args):
    from aiofranka.ipc import pid_file_for_ip

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol

    # Header
    ver = _get_version()
    print(f"\n  {BOLD}aiofranka{RST} {DIM}v{ver}{RST}  {DIM}|{RST}  {robot_ip}  {DIM}({protocol}){RST}\n")

    # Check if already running
    pid_path = pid_file_for_ip(robot_ip)
    if os.path.exists(pid_path):
        with open(pid_path) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            print(f"  {YELLOW}Already running{RST} {DIM}(PID {pid}){RST}")
            print(f"  Stop with: {BOLD}aiofranka stop{RST}\n")
            return
        except ProcessLookupError:
            os.unlink(pid_path)

    unlock = not args.no_unlock
    if unlock:
        username, password = _resolve_credentials(args)
    else:
        username, password = args.username, args.password

    # Pre-check: if robot is already unlocked with FCI active but token is held,
    # give the user a choice before launching the daemon.
    skip_token = False
    if unlock:
        skip_token = _check_control_token(robot_ip, username, password, protocol)

    lock_on_error = args.lock_on_error
    home = not args.no_home

    controller_cls = "native" if args.native else None
    if args.foreground:
        from aiofranka.server import run_server
        run_server(robot_ip, foreground=True, unlock=unlock,
                   username=username, password=password, protocol=protocol,
                   skip_token=skip_token, lock_on_error=lock_on_error,
                   home=home, controller_cls=controller_cls)
    else:
        from aiofranka.server import daemonize_and_run
        daemonize_and_run(robot_ip, unlock=unlock,
                          username=username, password=password, protocol=protocol,
                          skip_token=skip_token, lock_on_error=lock_on_error,
                          home=home, controller_cls=controller_cls)
        _wait_for_server(robot_ip)


def cmd_gravcomp(args):
    from aiofranka.server import _DeskClientV2, _clear_token, run_gravcomp_loop

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)
    damping = args.damping

    if args.mode == "program":
        # Hand guiding in Programming mode: the robot moves the arm while the guiding button
        # on the end effector is held, so no control loop runs here.
        print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} gravcomp --mode program {DIM}({robot_ip}){RST}\n")
        pid = _check_server_running(robot_ip)
        if pid is not None:
            print(f"  {RED}Error:{RST} the server (PID {pid}) is controlling the robot.")
            print(f"  Stop it first with: {BOLD}aiofranka stop{RST}\n")
            return
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
        if _change_mode(client, robot_ip, "Programming"):
            print(f"\n  {GREEN}Programming mode{RST}: hold the guiding button near the end effector to "
                  f"move the arm.")
            print(f"  For FCI: {BOLD}aiofranka unlock{RST}")
        print()
        return

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} gravcomp {DIM}({robot_ip}){RST}")
    print(f"  {DIM}kp=0  kd={damping}  (Ctrl+C stops control; joints stay unlocked){RST}\n")

    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
        _cli_take_control(client, robot_ip)

        print(f"\n  {GREEN}Running{RST} {DIM}— robot is in gravity compensation mode{RST}")
        print(f"  {DIM}You can freely move the robot by hand. Press Ctrl+C to stop.{RST}\n")

        # Run the control loop (blocks until Ctrl+C)
        run_gravcomp_loop(robot_ip, damping=damping, http_port=args.http_port)

        # Teardown — leave robot unlocked with FCI active, just release token
        print()  # blank line after ^C
        try:
            _cli_run_with_spinner("Releasing control token", 1, 1,
                                  client.release_token)
            _clear_token(robot_ip)
        except Exception:
            pass

        print(f"\n  {GREEN}Stopped{RST} {DIM}(joints left unlocked, FCI active — use {RST}{BOLD}aiofranka lock{RST}{DIM} to lock){RST}\n")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}\n")


def cmd_home(args):
    from aiofranka.server import _DeskClientV2, _clear_token, run_home_move

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} home {DIM}({robot_ip}){RST}\n")

    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
        _cli_take_control(client, robot_ip)

        print()
        run_home_move(robot_ip)

        try:
            _cli_run_with_spinner("Releasing control token", 1, 1,
                                  client.release_token)
            _clear_token(robot_ip)
        except Exception:
            pass

        print(f"\n  {GREEN}Done{RST} {DIM}(joints left unlocked, FCI active — use {RST}{BOLD}aiofranka lock{RST}{DIM} to lock){RST}\n")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}\n")


def _wait_for_server(robot_ip: str):
    """Poll daemon progress and render animated step output."""
    from aiofranka.ipc import (
        StateBlock, STATUS_RUNNING, STATUS_ERROR,
        progress_file_for_ip, pid_file_for_ip,
    )

    progress_path = progress_file_for_ip(robot_ip)
    spin_idx = 0
    cur_step = 0
    cur_total = 0
    # Pre-populate labels so steps that complete before first poll still render
    labels = {
        1: "Logging in to Franka Desk",
        2: "Acquiring control token",
        3: "Unlocking joints",
        4: "Activating FCI",
        5: "Starting 1kHz control loop",
        6: "Moving to home pose",
        7: "Starting 1kHz control loop",
    }
    got_progress = False
    no_token = False  # set True if server couldn't acquire control token

    for _ in range(600):   # 60s at 0.1s intervals
        time.sleep(0.1)
        spin_idx += 1
        frame = _SPINNER[spin_idx % len(_SPINNER)]

        # ── Read structured progress from daemon ──
        step, total, label, warn = 0, 0, "", ""
        try:
            with open(progress_path) as f:
                prog = json.load(f)
            step, total, label = prog["step"], prog["total"], prog["label"]
            warn = prog.get("warn", "")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        if warn == "no_token":
            no_token = True

        # ── Check shared memory for running / error ──
        srv_status = None
        err_msg = ""
        try:
            shm = StateBlock(robot_ip, create=False, track=False)
            srv_status = shm.read_status()
            if srv_status == STATUS_ERROR:
                err_msg = shm.read_error()
            shm.close()
        except FileNotFoundError:
            pass

        if srv_status == STATUS_RUNNING:
            # Print final step as done
            if got_progress and cur_step > 0:
                sys.stdout.write(
                    f"\r{_step_line(cur_step, cur_total, labels.get(cur_step, label), f'{GREEN}done{RST}')}\n"
                )
            elif not got_progress:
                sys.stdout.write(f"\r{' ' * 60}\r")
            sys.stdout.flush()

            pid = "?"
            try:
                with open(pid_file_for_ip(robot_ip)) as f:
                    pid = f.read().strip()
            except Exception:
                pass

            # Read self-test remaining from progress data
            self_test_remaining = 0
            try:
                with open(progress_path) as f:
                    final_prog = json.load(f)
                self_test_remaining = final_prog.get("self_test_remaining", 0)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            print(f"\n  {GREEN}Ready{RST} {DIM}(PID {pid}){RST}")
            print(f"  The robot is holding its position with a soft impedance")
            print(f"  controller {DIM}(kp=80, kd=4){RST}. You can gently push it to feel")
            print(f"  the compliance.")
            if self_test_remaining > 0:
                hours_left = self_test_remaining / 3600
                if hours_left >= 1:
                    print(f"\n  Self-test due in {DIM}{hours_left:.1f}h{RST}")
                else:
                    minutes_left = self_test_remaining / 60
                    print(f"\n  {YELLOW}Self-test due in {minutes_left:.0f}m{RST}")
            if no_token:
                print(f"\n  {YELLOW}Warning:{RST} Running without control token.")
                print(f"  The robot will {BOLD}not{RST} be locked when you stop the server.")
                print(f"  Lock it yourself via the Franka Desk web GUI when done.")
            print(f"\n  Stop with: {BOLD}aiofranka stop{RST}\n")
            return

        if srv_status == STATUS_ERROR:
            # Use progress file to figure out which step failed
            fail_step = step if step > 0 else cur_step
            fail_total = total if total > 0 else cur_total
            if fail_total == 0:
                fail_total = 5  # assume full unlock flow
            # Clear spinner line
            sys.stdout.write(f"\r{' ' * 60}\r")
            sys.stdout.flush()
            # Print completed steps before the failed one
            for s in range(1, fail_step):
                sys.stdout.write(f"{_step_line(s, fail_total, labels.get(s, '...'), f'{GREEN}done{RST}')}\n")
            # Print the failed step
            if fail_step > 0:
                sys.stdout.write(
                    f"{_step_line(fail_step, fail_total, labels.get(fail_step, label), f'{RED}failed{RST}')}\n"
                )
            sys.stdout.flush()
            print(f"\n  {RED}Error:{RST}")
            print(_wrap(err_msg))
            print(f"\n  Logs: {DIM}~/.aiofranka/server.log{RST}\n")
            return

        # ── No progress yet — show waiting spinner ──
        if step == 0:
            sys.stdout.write(f"\r  {YELLOW}{frame}{RST} Starting daemon...")
            sys.stdout.flush()
            continue

        # ── First progress arrived ──
        labels[step] = label
        if not got_progress:
            sys.stdout.write(f"\r{' ' * 60}\r")
            sys.stdout.flush()
            got_progress = True
            cur_total = total
            # Print all steps that already completed before we started polling
            for s in range(1, step):
                st = f"{YELLOW}skipped{RST}" if (no_token and s == 2) else f"{GREEN}done{RST}"
                sys.stdout.write(f"{_step_line(s, total, labels.get(s, '...'), st)}\n")
            sys.stdout.flush()
            cur_step = step

        # ── Step advanced — print completed steps ──
        if step > cur_step:
            for s in range(cur_step, step):
                st = f"{YELLOW}skipped{RST}" if (no_token and s == 2) else f"{GREEN}done{RST}"
                s_label = labels.get(s, "...")
                sys.stdout.write(f"\r{_step_line(s, total, s_label, st)}\n")
            sys.stdout.flush()
            cur_step = step
            cur_total = total
            labels[step] = label

        # ── Animate current step ──
        sys.stdout.write(f"\r{_step_line(step, total, label, f'{YELLOW}{frame}{RST}')}")
        sys.stdout.flush()

    # Timeout
    sys.stdout.write("\n")
    print(f"\n  {YELLOW}Timeout.{RST} Check: {DIM}~/.aiofranka/server.log{RST}\n")


def cmd_stop(args):
    from aiofranka.ipc import pid_file_for_ip, progress_file_for_ip

    robot_ip = _resolve_ip(args.ip)
    pid_path = pid_file_for_ip(robot_ip)

    if not os.path.exists(pid_path):
        print(f"\n  No server running for {robot_ip}\n")
        return

    with open(pid_path) as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        os.unlink(pid_path)
        print(f"\n  No server running for {robot_ip} {DIM}(stale PID file removed){RST}\n")
        return

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} stopping {robot_ip} {DIM}(PID {pid}){RST}\n")
    os.kill(pid, signal.SIGTERM)

    # Pre-populate labels for shutdown steps
    stop_labels = {
        1: "Stopping 1kHz control loop",
        2: "Deactivating FCI",
        3: "Locking joints",
        4: "Releasing control token",
    }
    progress_path = progress_file_for_ip(robot_ip)
    spin_idx = 0
    cur_step = 0
    cur_total = 0
    got_progress = False
    no_token = False

    # Check if server was running without control token (from startup progress)
    try:
        with open(progress_path) as f:
            prog = json.load(f)
        if prog.get("warn") == "no_token":
            no_token = True
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    for _ in range(300):   # 30s at 0.1s
        time.sleep(0.1)
        spin_idx += 1
        frame = _SPINNER[spin_idx % len(_SPINNER)]

        # Read shutdown progress from daemon
        step, total, label = 0, 0, ""
        try:
            with open(progress_path) as f:
                prog = json.load(f)
            step, total, label = prog["step"], prog["total"], prog["label"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        # Check if process exited
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # Re-read progress file one last time (daemon leaves it for us)
            final_step, final_total = 0, 0
            try:
                with open(progress_path) as f:
                    prog = json.load(f)
                final_step = prog.get("step", 0)
                final_total = prog.get("total", 0)
                final_label = prog.get("label", "")
                if final_total > 0:
                    stop_labels[final_step] = final_label
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

            # If we missed all shutdown progress, reconstruct from final state
            if not got_progress and final_total > 0 and final_step > 0:
                sys.stdout.write(f"\r{' ' * 60}\r")
                for s in range(1, final_total + 1):
                    sys.stdout.write(
                        f"{_step_line(s, final_total, stop_labels.get(s, '...'), f'{GREEN}done{RST}')}\n"
                    )
            elif got_progress and cur_step > 0:
                sys.stdout.write(
                    f"\r{_step_line(cur_step, cur_total, stop_labels.get(cur_step, label), f'{GREEN}done{RST}')}\n"
                )
                for s in range(cur_step + 1, cur_total + 1):
                    sys.stdout.write(
                        f"{_step_line(s, cur_total, stop_labels.get(s, '...'), f'{GREEN}done{RST}')}\n"
                    )
            elif not got_progress:
                sys.stdout.write(f"\r{' ' * 60}\r")
            sys.stdout.flush()

            # Clean up progress file (daemon left it for us)
            try:
                os.unlink(progress_path)
            except FileNotFoundError:
                pass

            print(f"\n  {GREEN}Stopped{RST}")
            if no_token:
                print(f"\n  {YELLOW}Warning:{RST} Joints were {BOLD}not{RST} locked (no control token).")
                print(f"  Lock them yourself via the Franka Desk web GUI.")
            print()
            return

        # No shutdown progress yet — show waiting spinner
        if step == 0 or not label.startswith("Stop"):
            # Still waiting for shutdown to begin (progress file has startup data)
            sys.stdout.write(f"\r  {YELLOW}{frame}{RST} Waiting for shutdown...")
            sys.stdout.flush()
            continue

        # Record label
        stop_labels[step] = label

        # First shutdown progress
        if not got_progress:
            sys.stdout.write(f"\r{' ' * 60}\r")
            sys.stdout.flush()
            got_progress = True
            cur_total = total
            for s in range(1, step):
                sys.stdout.write(f"{_step_line(s, total, stop_labels.get(s, '...'), f'{GREEN}done{RST}')}\n")
            sys.stdout.flush()
            cur_step = step

        # Step advanced
        if step > cur_step:
            for s in range(cur_step, step):
                sys.stdout.write(f"\r{_step_line(s, total, stop_labels.get(s, '...'), f'{GREEN}done{RST}')}\n")
            sys.stdout.flush()
            cur_step = step
            cur_total = total

        # Animate current step
        sys.stdout.write(f"\r{_step_line(step, total, label, f'{YELLOW}{frame}{RST}')}")
        sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"\n  {YELLOW}Still running after 30s.{RST} Kill manually: kill -9 {pid}\n")


def cmd_status(args):
    from aiofranka.ipc import pid_file_for_ip

    robot_ip = _resolve_ip(args.ip)
    pid_path = pid_file_for_ip(robot_ip)
    protocol = args.protocol

    ver = _get_version()
    print(f"\n  {BOLD}aiofranka{RST} {DIM}v{ver}{RST}  {DIM}|{RST}  {robot_ip}  {DIM}({protocol}){RST}")

    # ── Server status ──
    pid = None
    if os.path.exists(pid_path):
        with open(pid_path) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            os.unlink(pid_path)
            pid = None

    print()
    if pid is not None:
        print(f"  {BOLD}Server{RST}")
        print(f"    Status ........... {GREEN}running{RST} {DIM}(PID {pid}){RST}")

        try:
            import msgpack
            import zmq
            from aiofranka.ipc import zmq_endpoint_for_ip

            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            sock.connect(zmq_endpoint_for_ip(robot_ip))
            sock.send(msgpack.packb({"cmd": "status"}, use_bin_type=True))
            resp = msgpack.unpackb(sock.recv(), raw=False)
            ctrl = resp.get("controller_type", "?")
            print(f"    Controller ....... {ctrl}")
            sock.close()
            ctx.term()
        except Exception:
            pass
    else:
        print(f"  {BOLD}Server{RST}")
        print(f"    Status ........... {DIM}not running{RST}")

    # ── Robot status from Desk API ──
    username, password = None, None
    config = _load_config()
    username = config.get("username")
    password = config.get("password")

    if not username or not password:
        # No saved credentials — skip robot queries
        print()
        return

    try:
        from aiofranka.server import _DeskClientV2, _load_token_state

        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)

        # System state
        sys_state = client.get_system_state()
        sys_status = sys_state.get("status", "")
        serial = sys_state.get("controlSerialNumber", "")

        print()
        print(f"  {BOLD}System{RST}")
        if sys_status:
            color = GREEN if sys_status == "Started" else YELLOW
            print(f"    Status ........... {color}{sys_status}{RST}")
        if serial:
            print(f"    Serial ........... {DIM}{serial}{RST}")

        op_mode = sys_state.get("operatingMode", {}).get("status", "") or client.get_operating_mode()
        if op_mode:
            color = GREEN if op_mode == "Execution" else YELLOW
            print(f"    Operating mode ... {color}{op_mode}{RST}")

        # Arm info
        arm_info = client.get_arm_info()
        arm_status = arm_info.get("status", "")
        arm_type = arm_info.get("armType", "")
        if arm_status:
            color = GREEN if arm_status == "Connected" else YELLOW
            label = f"{arm_status}"
            if arm_type:
                label += f" ({arm_type})"
            print(f"    Arm .............. {color}{label}{RST}")

        # Joints
        print()
        print(f"  {BOLD}Robot{RST}")
        joints_unlocked = client.are_joints_unlocked()
        if joints_unlocked:
            print(f"    Joints ........... {GREEN}unlocked{RST}")
        else:
            print(f"    Joints ........... {DIM}locked{RST}")

        # FCI
        fci_active = client.is_fci_active()
        if fci_active:
            print(f"    FCI .............. {GREEN}active{RST}")
        else:
            print(f"    FCI .............. {DIM}inactive{RST}")

        # Control token (new API)
        token_r = client._req("GET", "/api/system/control-token",
                              headers=client._headers())
        if token_r.status_code == 200:
            token_data = token_r.json()
            owner = token_data.get("owner")
            if owner:
                print(f"    Control token .... {YELLOW}held{RST} {DIM}(owner: {owner}){RST}")
            else:
                print(f"    Control token .... {DIM}free{RST}")

        # Saved standalone token
        saved_token, saved_token_id = _load_token_state(robot_ip)
        if saved_token:
            print(f"    Saved token ...... {DIM}{saved_token[:20]}...{RST}")

        # Self-test
        st = client.get_self_test_status()
        st_status = st.get("status", "")
        st_remaining = st.get("remaining", 0)
        if st_status:
            if st_status == "Elapsed":
                st_text = f"{RED}overdue{RST}"
            elif st_status == "Warning":
                hours = st_remaining / 3600
                st_text = f"{YELLOW}due soon{RST} {DIM}({hours:.1f}h remaining){RST}"
            elif st_status == "OK":
                hours = st_remaining / 3600
                st_text = f"{GREEN}ok{RST} {DIM}({hours:.1f}h remaining){RST}"
            elif st_status == "Running":
                st_text = f"{YELLOW}running{RST}"
            else:
                st_text = st_status
            print(f"    Self-test ........ {st_text}")

        # End-effector configuration
        sys_config = client.get_configuration()
        ee_config = sys_config.get("endEffectorConfiguration", {})
        ee = ee_config.get("endEffector", {})
        if ee:
            print()
            print(f"  {BOLD}End Effector{RST}")
            ee_name = ee.get("name", "?")
            # The configuration names only the kind ("Other"); the active Desk profile has the tool's name.
            try:
                from aiofranka.tools import list_tools

                active = next((t for t in list_tools(robot_ip, username, password, protocol).values()
                               if t.active), None)
            except Exception:
                active = None
            if active is not None:
                device = "Franka Hand" if active.device == "ee-gripper" else "Generic Device"
                print(f"    Tool ............. {active.name}  {DIM}{device}{RST}")
            else:
                print(f"    Type ............. {ee_name}")

            params = ee.get("params", {})
            if "mass" in params:
                print(f"    Mass ............. {params['mass']:.3f} kg")

            com = params.get("centerOfMass", {})
            if com:
                print(f"    CoM .............. [{com.get('x', 0):.4f}, {com.get('y', 0):.4f}, {com.get('z', 0):.4f}] m")

            inertia = params.get("inertia", {})
            if inertia:
                i11 = inertia.get("x11", 0)
                i12 = inertia.get("x12", 0)
                i13 = inertia.get("x13", 0)
                i22 = inertia.get("x22", 0)
                i23 = inertia.get("x23", 0)
                i33 = inertia.get("x33", 0)
                print(f"    Inertia .......... [{i11:.4f}, {i12:.4f}, {i13:.4f}] kg*m^2")
                print(f"                       [{i12:.4f}, {i22:.4f}, {i23:.4f}]")
                print(f"                       [{i13:.4f}, {i23:.4f}, {i33:.4f}]")

            tf = params.get("transformation", {})
            if tf:
                t = tf.get("translation", {})
                r = tf.get("rotation", {})
                if t:
                    print(f"    Translation ...... [{t.get('x', 0):.4f}, {t.get('y', 0):.4f}, {t.get('z', 0):.4f}] m")
                if r:
                    print(f"    Rotation (RPY) ... [{r.get('roll', 0):.4f}, {r.get('pitch', 0):.4f}, {r.get('yaw', 0):.4f}] rad")

    except Exception as e:
        print(f"    {DIM}(could not query robot: {e}){RST}")

    print()

def _change_mode(client, robot_ip: str, desired: str) -> bool:
    """
    Change the operating mode to desired ("Programming" or "Execution") as Desk's mode
    switch does, printing the steps, and hand the control token back to Desk. Programming
    deactivates FCI and opens the brakes for hand guiding. Returns whether it succeeded.
    """
    from aiofranka.server import _load_token_state, _clear_token

    change = client.get_operating_mode() != desired
    # Programming: FCI only runs in Execution, and hand guiding needs open brakes.
    deactivate_fci = desired == "Programming" and client.is_fci_active()
    unlock_joints = desired == "Programming" and not client.are_joints_unlocked()
    total = 2 + int(change) + int(deactivate_fci) + int(unlock_joints)
    try:
        # Reuse the token aiofranka unlock saved: taking another would wait for it.
        step = 1
        client._token, client._token_id = _load_token_state(robot_ip)
        if client._token is not None and client.validate_token():
            print(_cli_step_line(step, total, "Acquiring control token",
                                 f"{GREEN}done{RST} {DIM}(reused){RST}"))
        else:
            if client._token is not None:  # stale: release it so take_token won't deadlock
                client.release_token(best_effort=True)
            client._token = client._token_id = None
            _cli_run_with_spinner("Acquiring control token", step, total,
                                  client.take_token, timeout=15)

        if deactivate_fci:
            step += 1
            _cli_run_with_spinner("Deactivating FCI", step, total, client.deactivate_fci)
        if change:
            step += 1
            _cli_run_with_spinner(f"Changing operating mode to {desired}", step, total,
                                  client.change_operating_mode, desired)
        if unlock_joints:
            step += 1
            _cli_run_with_spinner("Unlocking joints", step, total, client.unlock)
        return True
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}")
        return False
    finally:
        # Hand control back to Desk; aiofranka unlock takes it again for FCI.
        if client._token is not None:
            _cli_run_with_spinner("Releasing control token", total, total,
                                  client.release_token, best_effort=True)
            _clear_token(robot_ip)


def cmd_mode(args):
    """View or change the operating mode."""
    from aiofranka.server import _DeskClientV2

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)

    client = _DeskClientV2(robot_ip, username, password, protocol=protocol)

    if args.mode:
        desired = {"program": "Programming", "execute": "Execution"}[args.mode]
        if client.get_operating_mode() == desired:
            print(f"\n  Already in {desired}.\n")
            return
        pid = _check_server_running(robot_ip)
        if pid is not None:
            print(f"\n  {RED}Error:{RST} the server (PID {pid}) is controlling the robot.")
            print(f"  Stop it first with: {BOLD}aiofranka stop{RST}\n")
            return

        print()
        if _change_mode(client, robot_ip, desired):
            print(f"\n  {GREEN}Operating mode changed to {desired}{RST}")
            if desired == "Programming":
                print(f"  Hand-guide with the guiding button near the end effector.")
            print(f"  For FCI: {BOLD}aiofranka unlock{RST}")
        print()
    else:
        # Just display current mode
        op_mode = client.get_operating_mode()
        sys_state = client.get_system_state()

        print()
        print(f"  {BOLD}Operating Mode{RST}")

        if op_mode:
            color = GREEN if op_mode == "Execution" else YELLOW
            print(f"    Current .......... {color}{op_mode}{RST}")
        else:
            print(f"    Current .......... {DIM}unknown{RST}")

        sys_status = sys_state.get("status", "")
        if sys_status:
            color = GREEN if sys_status == "Started" else YELLOW
            print(f"    System status .... {color}{sys_status}{RST}")

        print()
        print(f"  {BOLD}Available Modes{RST}")
        _exec = f"{GREEN}Execution{RST}" if op_mode == "Execution" else "Execution"
        _prog = f"{GREEN}Programming{RST}" if op_mode == "Programming" else "Programming"
        print(f"    {_exec:30s} {DIM}FCI / programmatic control{RST}")
        print(f"    {_prog:30s} {DIM}freedrive via button near end-effector{RST}")
        print()
        if op_mode == "Execution":
            print(f"  {DIM}Tip: aiofranka mode program switches to Programming, to freedrive")
            print(f"  the robot by holding the button on the pilot interface near the end-effector.{RST}")
        else:
            print(f"  {DIM}aiofranka unlock (or aiofranka mode execute) switches back for FCI control.{RST}")
        print()

def cmd_config(args):
    from aiofranka.server import _DeskClientV2, _load_token_state

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} config {DIM}({robot_ip}){RST}\n")

    client = _DeskClientV2(robot_ip, username, password, protocol=protocol)

    # If no flags given, just print current config and exit
    has_changes = any([
        args.mass is not None,
        args.com is not None,
        args.inertia is not None,
        args.translation is not None,
        args.rotation is not None,
        args.ee_name is not None,
    ])

    if not has_changes:
        # Show current config
        sys_config = client.get_configuration()
        ee_config = sys_config.get("endEffectorConfiguration", {})
        ee = ee_config.get("endEffector", {})
        if not ee:
            print(f"  {DIM}No end-effector configuration found.{RST}\n")
            return
        print(f"  {BOLD}End Effector{RST}")
        print(f"    name: {ee.get('name', '?')}")
        params = ee.get("params", {})
        if "mass" in params:
            print(f"    mass: {params['mass']}")
        com = params.get("centerOfMass", {})
        if com:
            print(f"    com: {com.get('x', 0)}, {com.get('y', 0)}, {com.get('z', 0)}")
        inertia = params.get("inertia", {})
        if inertia:
            print(f"    inertia: {inertia.get('x11', 0)}, {inertia.get('x12', 0)}, {inertia.get('x13', 0)}, "
                  f"{inertia.get('x22', 0)}, {inertia.get('x23', 0)}, {inertia.get('x33', 0)}")
        tf = params.get("transformation", {})
        t = tf.get("translation", {})
        r = tf.get("rotation", {})
        if t:
            print(f"    translation: {t.get('x', 0)}, {t.get('y', 0)}, {t.get('z', 0)}")
        if r:
            print(f"    rotation: {r.get('roll', 0)}, {r.get('pitch', 0)}, {r.get('yaw', 0)}")
        print(f"\n  Set values with flags, e.g.:")
        print(f"    {BOLD}aiofranka config --mass 0.5 --com 0,0,0.03{RST}\n")
        return

    # Build the EE params patch — start from current config so we only override what's specified
    sys_config = client.get_configuration()
    ee_config = sys_config.get("endEffectorConfiguration", {})
    cur_ee = ee_config.get("endEffector", {})
    cur_params = cur_ee.get("params", {})

    ee_name = args.ee_name if args.ee_name is not None else cur_ee.get("name", "Other")
    params = {}

    # Mass
    if args.mass is not None:
        params["mass"] = args.mass
    elif "mass" in cur_params:
        params["mass"] = cur_params["mass"]

    # CoM
    if args.com is not None:
        vals = [float(v) for v in args.com.split(",")]
        if len(vals) != 3:
            print(f"  {RED}Error:{RST} --com expects 3 values: x,y,z\n")
            return
        params["centerOfMass"] = {"x": vals[0], "y": vals[1], "z": vals[2]}
    elif "centerOfMass" in cur_params:
        params["centerOfMass"] = cur_params["centerOfMass"]

    # Inertia (upper triangle: x11,x12,x13,x22,x23,x33)
    if args.inertia is not None:
        vals = [float(v) for v in args.inertia.split(",")]
        if len(vals) != 6:
            print(f"  {RED}Error:{RST} --inertia expects 6 values: x11,x12,x13,x22,x23,x33\n")
            return
        params["inertia"] = {
            "x11": vals[0], "x12": vals[1], "x13": vals[2],
            "x22": vals[3], "x23": vals[4], "x33": vals[5],
        }
    elif "inertia" in cur_params:
        params["inertia"] = cur_params["inertia"]

    # Transformation
    tf = cur_params.get("transformation", {})
    if args.translation is not None:
        vals = [float(v) for v in args.translation.split(",")]
        if len(vals) != 3:
            print(f"  {RED}Error:{RST} --translation expects 3 values: x,y,z\n")
            return
        tf["translation"] = {"x": vals[0], "y": vals[1], "z": vals[2]}
    if args.rotation is not None:
        vals = [float(v) for v in args.rotation.split(",")]
        if len(vals) != 3:
            print(f"  {RED}Error:{RST} --rotation expects 3 values: roll,pitch,yaw\n")
            return
        tf["rotation"] = {"roll": vals[0], "pitch": vals[1], "yaw": vals[2]}
    if tf:
        params["transformation"] = tf

    patch = {
        "endEffectorConfiguration": {
            "endEffector": {
                "name": ee_name,
                "params": params,
            }
        }
    }

    # Acquire token
    saved_token, saved_token_id = _load_token_state(robot_ip)
    took_token = False
    if saved_token is not None:
        client._token = saved_token
        client._token_id = saved_token_id
        if not client.validate_token():
            client._token = None
            client._token_id = None
    if client._token is None:
        try:
            client.take_token(timeout=15)
            took_token = True
        except RuntimeError as e:
            print(f"  {RED}Error:{RST} Could not acquire control token: {e}\n")
            return

    try:
        client.set_configuration(patch)
        print(f"  {GREEN}End-effector configuration updated{RST}")

        # Show new config
        new_config = client.get_configuration()
        new_ee = new_config.get("endEffectorConfiguration", {}).get("endEffector", {})
        new_params = new_ee.get("params", {})
        print(f"    Name ............. {new_ee.get('name', '?')}")
        if "mass" in new_params:
            print(f"    Mass ............. {new_params['mass']:.3f} kg")
        com = new_params.get("centerOfMass", {})
        if com:
            print(f"    CoM .............. [{com.get('x', 0):.4f}, {com.get('y', 0):.4f}, {com.get('z', 0):.4f}] m")
        inertia = new_params.get("inertia", {})
        if inertia:
            print(f"    Inertia .......... [{inertia.get('x11', 0):.4f}, {inertia.get('x12', 0):.4f}, {inertia.get('x13', 0):.4f}] kg*m^2")
            print(f"                       [{inertia.get('x12', 0):.4f}, {inertia.get('x22', 0):.4f}, {inertia.get('x23', 0):.4f}]")
            print(f"                       [{inertia.get('x13', 0):.4f}, {inertia.get('x23', 0):.4f}, {inertia.get('x33', 0):.4f}]")
        new_tf = new_params.get("transformation", {})
        t = new_tf.get("translation", {})
        r = new_tf.get("rotation", {})
        if t:
            print(f"    Translation ...... [{t.get('x', 0):.4f}, {t.get('y', 0):.4f}, {t.get('z', 0):.4f}] m")
        if r:
            print(f"    Rotation (RPY) ... [{r.get('roll', 0):.4f}, {r.get('pitch', 0):.4f}, {r.get('yaw', 0):.4f}] rad")
        print()
    except Exception as e:
        print(f"  {RED}Error:{RST} {e}\n")
    finally:
        if took_token:
            try:
                client.release_token(best_effort=True)
            except Exception:
                pass


def _print_tool(tool):
    def fmt(values):
        return ", ".join(f"{v:.4f}" for v in values)

    mark = f" {GREEN}(active){RST}" if tool.active else ""
    device = "Franka Hand" if tool.device == "ee-gripper" else "Generic Device"
    print(f"  {BOLD}{tool.name}{RST}{mark}  {DIM}{device}{RST}")
    print(f"    Mass ............. {tool.mass:.3f} kg")
    print(f"    CoM .............. [{fmt(tool.com)}] m")
    print(f"    Inertia (diag) ... [{fmt(tool.inertia.diagonal())}] kg*m^2")
    print(f"    TCP .............. [{fmt(tool.translation)}] m, RPY [{fmt(tool.rotation)}] rad")


def _cli_take_control(client, robot_ip: str):
    """Take the control token, recover errors, switch from Programming to Execution, unlock
    the joints and activate FCI."""
    from aiofranka.server import _load_token_state, _save_token_state, _clear_token

    # FCI only activates in Execution, e.g. not after hand guiding in Programming.
    programming = client.get_operating_mode() == "Programming"
    total = 4 + int(programming)
    saved_token, saved_token_id = _load_token_state(robot_ip)
    if saved_token is not None:
        client._token = saved_token
        client._token_id = saved_token_id
        if not client.validate_token():
            client._token = None
            client._token_id = None

    if client._token is None:
        _cli_run_with_spinner("Acquiring control token", 1, total, client.take_token, timeout=15)
    else:
        print(_cli_step_line(1, total, "Acquiring control token", f"{GREEN}done{RST} {DIM}(reused){RST}"))

    try:
        _cli_run_with_spinner("Recovering safety errors", 2, total, client.recover_errors)
        step = 3
        if programming:
            _cli_run_with_spinner("Changing operating mode to Execution", step, total,
                                  client.change_operating_mode, "Execution")
            step += 1
        if client.are_joints_unlocked():
            print(_cli_step_line(step, total, "Unlocking joints", f"{GREEN}done{RST} {DIM}(already){RST}"))
        else:
            _cli_run_with_spinner("Unlocking joints", step, total, client.unlock)
        if client.is_fci_active():
            print(_cli_step_line(step + 1, total, "Activating FCI", f"{GREEN}done{RST} {DIM}(already){RST}"))
        else:
            _cli_run_with_spinner("Activating FCI", step + 1, total, client.activate_fci)
        _save_token_state(robot_ip, client._token, client._token_id)
    except Exception:
        try:
            client.release_token(best_effort=True)
            _clear_token(robot_ip)
        except Exception:
            pass
        raise


async def _identify_tool(robot_ip: str, kwargs: dict):
    """Connect and identify the tool on the flange; identify_payload() starts and stops the loop."""
    import numpy as np
    from aiofranka.native import NativeFrankaController
    from aiofranka.robot import RobotInterface

    # The 1 kHz loop runs in C++, so nothing in Python delays a torque command.
    controller = NativeFrankaController(RobotInterface(robot_ip, read_tool=False))
    # Joint impedance gains as for moving home.
    gains = np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])
    controller.kp, controller.kd = gains * 80, gains * 4
    return await controller.identify_payload(**kwargs)


def _ask_number(label: str, value: float) -> float:
    """Ask for a number, keeping value on Enter."""
    while True:
        text = input(f"    {label} [{value:g}]: ").strip()
        if not text:
            return value
        try:
            return float(text)
        except ValueError:
            print(f"    {RED}Not a number{RST}")


def _confirm_tool(name: str, mass: float, com, activate: bool):
    """
    Ask whether to save the tool, offering to edit its mass and center of mass.

    Returns the (mass [kg], com [m]) to save, or None to not save.
    """
    import numpy as np

    action = "Save and activate" if activate else "Save"
    print()
    while True:
        answer = input(f"  {action} it as {BOLD}{name}{RST} in Desk? [Y]es, [e]dit, [n]o: ").strip().lower()
        if answer in ("", "y", "yes"):
            return mass, com
        if answer in ("n", "no"):
            return None
        if answer in ("e", "edit"):
            print(f"  {DIM}Enter keeps a value.{RST}")
            mass = _ask_number("Mass [g]", round(1e3 * mass, 1)) / 1e3
            com = np.array([_ask_number(f"Center of mass {axis} [mm]", round(1e3 * c, 1))
                            for axis, c in zip("xyz", com)]) / 1e3
            mm = ", ".join(f"{axis} {1e3 * c:+.1f}" for axis, c in zip("xyz", com))
            print(f"\n  {1e3 * mass:.1f} g, center of mass {mm} mm")


def cmd_tool(args):
    import asyncio
    from aiofranka import tools

    command = args.tool_command
    try:
        robot_ip = _resolve_ip(args.ip)
        username, password = _resolve_credentials(args)
        desk = dict(ip=robot_ip, username=username, password=password, protocol=args.protocol)
        print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} tool {command} {DIM}({robot_ip}){RST}\n")

        if command == "list":
            for tool in tools.list_tools(**desk).values():
                _print_tool(tool)
            print()

        elif command == "load":
            tool = _cli_run_with_spinner(f"Activating {args.name}", 1, 1, tools.load_tool, args.name, **desk)
            print()
            _print_tool(tool)
            print(f"\n  {DIM}Programs connected to the robot pick it up when they reconnect.{RST}\n")

        elif command == "unload":
            tool = _cli_run_with_spinner("Activating No End Effector", 1, 1, tools.unload_tool, **desk)
            print()
            _print_tool(tool)
            print(f"\n  {DIM}Programs connected to the robot pick it up when they reconnect.{RST}\n")

        elif command == "remove":
            if not args.yes and input(f"  Delete {args.name} from Desk? [y/N] ").strip().lower() != "y":
                print()
                return
            _cli_run_with_spinner(f"Removing {args.name}", 1, 1, tools.remove_tool, args.name, **desk)
            print()

        elif command == "identify":
            from aiofranka.server import _DeskClientV2, _clear_token

            pid = _check_server_running(robot_ip)
            if pid is not None:
                print(f"  {RED}Error:{RST} A server (PID {pid}) is connected to the robot. "
                      f"Stop it with {BOLD}aiofranka stop{RST} first.\n")
                return

            client = _DeskClientV2(robot_ip, username, password, protocol=args.protocol)
            _cli_take_control(client, robot_ip)
            try:
                print()
                print(_wrap(
                    f"The robot will move through {args.poses} poses around the current one, "
                    "approaching each from both sides, checked for collisions with a "
                    f"{args.tool_length} m x {args.tool_radius} m cylinder around the tool and a "
                    f"floor at {args.floor} m. Start in an open pose and keep a hand on the e-stop."
                ))
                if not args.yes:
                    input(f"\n  Press Enter to start, Ctrl+C to cancel... ")
                print()
                estimate = asyncio.run(_identify_tool(robot_ip, dict(
                    n_poses=args.poses, speed=args.speed, tool_length=args.tool_length,
                    tool_radius=args.tool_radius, floor=args.floor,
                )))

                mass, com = estimate.mass, estimate.com
                if not args.yes:
                    chosen = _confirm_tool(args.name, mass, com, activate=not args.no_load)
                    if chosen is None:
                        print()
                        return
                    mass, com = chosen
                print()
                steps = 1 if args.no_load else 2
                tool = _cli_run_with_spinner(f"Saving {args.name}", 1, steps, tools.save_tool,
                                             args.name, mass, com, **desk)
                if not args.no_load:
                    tool = _cli_run_with_spinner(f"Activating {args.name}", 2, steps, tools.load_tool,
                                                 args.name, **desk)
            finally:
                try:
                    _cli_run_with_spinner("Releasing control token", 1, 1, client.release_token)
                    _clear_token(robot_ip)
                except Exception:
                    pass

            print()
            _print_tool(tool)
            print()
    except KeyboardInterrupt:
        print()
    except SystemExit:
        # The control loop exits when the robot aborts the motion; it printed why.
        print(f"\n  {RED}Error:{RST} The control loop stopped, nothing was saved.\n")
    except (KeyError, ValueError) as e:
        print(f"\n  {RED}Error:{RST} {e.args[0] if e.args else e}\n")
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}\n")


def cmd_camera(args):
    try:
        from aiofranka import camera
    except ImportError as e:
        print(f"\n  {RED}Error:{RST} {e}. Install the camera extra: "
              f"{BOLD}pip install 'aiofranka[camera]'{RST}\n")
        return
    from pathlib import Path

    try:
        if args.camera_command == "fit":
            print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} camera fit {DIM}({args.session}){RST}\n")
            _print_calibration(camera.fit_session(args.session), Path(args.session))
            return

        from aiofranka.server import _DeskClientV2

        robot_ip = _resolve_ip(args.ip)
        username, password = _resolve_credentials(args)
        print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} camera calibrate {DIM}({robot_ip}){RST}\n")
        pid = _check_server_running(robot_ip)
        if pid is not None:
            print(f"  {RED}Error:{RST} A server (PID {pid}) is connected to the robot. "
                  f"Stop it with {BOLD}aiofranka stop{RST} first.\n")
            return
        # Programming mode, where the guiding button on the end effector moves the arm.
        client = _DeskClientV2(robot_ip, username, password, protocol=args.protocol)
        if not _change_mode(client, robot_ip, "Programming"):
            print()
            return
        session = camera.calibrate(robot_ip, stream=args.stream, cube=args.cube or camera.CUBE,
                                   out=args.out, username=username, password=password,
                                   protocol=args.protocol)
        print(f"\n  The robot stays in Programming mode; {BOLD}aiofranka unlock{RST} switches back for FCI.")
        if (session / "calibration.json").exists():
            _print_calibration(json.loads((session / "calibration.json").read_text()), session)
        else:
            print(f"\n  Stopped before fitting. The views are in {session}; fit them with:\n"
                  f"    {BOLD}aiofranka camera fit {session}{RST}\n")
    except KeyboardInterrupt:
        print()
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}\n")


def _print_calibration(calibration, session):
    """Where the camera and the cube are, and how well the fit reprojects the views."""
    import numpy as np

    camera = np.array(calibration["T_base_camera"])
    cube = np.array(calibration["T_ee_cube"])
    metrics = calibration["metrics"]
    held_out, everything = metrics["held_out"], metrics["all_views"]
    worst = sorted(everything["per_view"], key=lambda view: -view["rms_px"])[:3]
    print(f"  Camera ......... at {np.round(camera[:3, 3], 3).tolist()} m, "
          f"looking along {np.round(camera[:3, 2], 2).tolist()} (base frame)")
    print(f"  Cube on flange . at {np.round(cube[:3, 3], 3).tolist()} m")
    print(f"  Error .......... {held_out['rms_px']:.2f} px on {held_out['view_count']} held-out views, "
          f"{everything['rms_px']:.2f} px on all {everything['view_count']}")
    print(f"  Worst views .... " + ", ".join(f"{view['view_index']}: {view['rms_px']:.2f} px" for view in worst))
    print(f"  Saved .......... {session / 'calibration.json'}")
    if held_out["rms_px"] > 2.0:
        print(f"\n  {YELLOW}The held-out error is above 2 px.{RST} Check the worst views' images, "
              "or capture more varied poses.")
    print()


def _check_server_running(robot_ip: str) -> int | None:
    """Return PID if a server is running for this IP, else None."""
    from aiofranka.ipc import pid_file_for_ip
    pid_path = pid_file_for_ip(robot_ip)
    if not os.path.exists(pid_path):
        return None
    with open(pid_path) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 0)
        return pid
    except ProcessLookupError:
        os.unlink(pid_path)
        return None


def _cli_step_line(step: int, total: int, label: str, status: str) -> str:
    tag = f"{DIM}[{step}/{total}]{RST}"
    ndots = max(2, 38 - len(label))
    dots = " " + "." * ndots
    return f"  {tag} {label}{dots} {status}"


def _cli_run_with_spinner(label: str, step: int, total: int, fn, *args, **kwargs):
    """Run fn() in a thread, showing a spinner on the current step."""
    import threading

    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()

    spin_idx = 0
    while t.is_alive():
        frame = _SPINNER[spin_idx % len(_SPINNER)]
        line = _cli_step_line(step, total, label, f"{YELLOW}{frame}{RST}")
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()
        t.join(timeout=0.1)
        spin_idx += 1

    if error[0] is not None:
        line = _cli_step_line(step, total, label, f"{RED}failed{RST}")
        sys.stdout.write(f"\r{line}\n")
        sys.stdout.flush()
        raise error[0]

    line = _cli_step_line(step, total, label, f"{GREEN}done{RST}")
    sys.stdout.write(f"\r{line}\n")
    sys.stdout.flush()
    return result[0]


def cmd_lock(args):
    from aiofranka.server import (
        _DeskClientV2, _load_token_state, _clear_token,
    )

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} lock {DIM}({robot_ip}){RST}\n")

    pid = _check_server_running(robot_ip)
    if pid is not None:
        print(f"  {YELLOW}Server is running{RST} {DIM}(PID {pid}){RST}")
        print(f"  Use {BOLD}aiofranka stop{RST} instead (it will lock joints automatically).\n")
        return

    total = 4
    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)

        # Step 1: Acquire control token (reuse saved token if valid)
        saved_token, saved_token_id = _load_token_state(robot_ip)
        if saved_token is not None:
            client._token = saved_token
            client._token_id = saved_token_id
            if not client.validate_token():
                # Saved token is stale — release it so take_token won't deadlock
                try:
                    client.release_token(best_effort=True)
                except Exception:
                    pass
                _clear_token(robot_ip)
                client._token = None
                client._token_id = None

        if client._token is None:
            _cli_run_with_spinner("Acquiring control token", 1, total,
                                  client.take_token, timeout=15)
        else:
            print(_cli_step_line(1, total, "Acquiring control token",
                                 f"{GREEN}done{RST} {DIM}(reused){RST}"))

        try:
            # Step 2: Deactivate FCI
            _cli_run_with_spinner("Deactivating FCI", 2, total,
                                  client.deactivate_fci)

            # Step 3: Lock joints
            _cli_run_with_spinner("Locking joints", 3, total, client.lock)

            # Step 4: Release token
            _cli_run_with_spinner("Releasing control token", 4, total,
                                  client.release_token)
            _clear_token(robot_ip)
        except Exception:
            try:
                client.release_token(best_effort=True)
                _clear_token(robot_ip)
            except Exception:
                pass
            raise

        print(f"\n  {GREEN}Locked{RST}\n")
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}\n")


def cmd_unlock(args):
    from aiofranka.server import (
        _DeskClientV2, _save_token_state, _load_token_state, _clear_token,
    )

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} unlock {DIM}({robot_ip}){RST}\n")

    pid = _check_server_running(robot_ip)
    if pid is not None:
        print(f"  {YELLOW}Server is running{RST} {DIM}(PID {pid}){RST}")
        print(f"  Robot is already unlocked with FCI active.\n")
        return

    total = 4
    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)

        # Overdue self-tests keep the joints locked, and FCI only activates in
        # Execution: run them and switch back from Programming (hand guiding) first.
        self_test_due = client.get_self_test_status().get("status") == "Elapsed"
        programming = client.get_operating_mode() == "Programming"
        if self_test_due:
            total += 1
        if programming:
            total += 1

        # Step 1: Acquire control token (reuse saved token if valid)
        saved_token, saved_token_id = _load_token_state(robot_ip)
        if saved_token is not None:
            client._token = saved_token
            client._token_id = saved_token_id
            if not client.validate_token():
                # Saved token is stale — release it so take_token won't deadlock
                try:
                    client.release_token(best_effort=True)
                except Exception:
                    pass
                _clear_token(robot_ip)
                client._token = None
                client._token_id = None

        if client._token is None:
            _cli_run_with_spinner("Acquiring control token", 1, total,
                                  client.take_token, timeout=15)
        else:
            print(_cli_step_line(1, total, "Acquiring control token",
                                 f"{GREEN}done{RST} {DIM}(reused){RST}"))

        try:
            # Step 2: Recover safety errors
            _cli_run_with_spinner("Recovering safety errors", 2, total,
                                  client.recover_errors)
            step = 3

            if self_test_due:
                _cli_run_with_spinner("Running self-tests (overdue)", step, total,
                                      client.execute_self_tests)
                step += 1

            if programming:
                _cli_run_with_spinner("Switching to Execution", step, total,
                                      client.change_operating_mode, "Execution")
                step += 1

            _cli_run_with_spinner("Unlocking joints", step, total, client.unlock)
            _cli_run_with_spinner("Activating FCI", step + 1, total,
                                  client.activate_fci)

            _save_token_state(robot_ip, client._token, client._token_id)
        except Exception:
            try:
                client.release_token(best_effort=True)
                _clear_token(robot_ip)
            except Exception:
                pass
            raise

        print(f"\n  {GREEN}Unlocked{RST} {DIM}(FCI active){RST}")
        print(f"  Lock with: {BOLD}aiofranka lock{RST}\n")
    except Exception as e:
        print(f"\n  {RED}Error:{RST} {e}\n")


def cmd_selftest(args):
    from aiofranka.server import _DeskClientV2, _load_token_state, _clear_token
    from aiofranka.ipc import pid_file_for_ip

    robot_ip = _resolve_ip(args.ip)
    protocol = args.protocol
    username, password = _resolve_credentials(args)

    print(f"\n  {BOLD}aiofranka{RST} {DIM}|{RST} self-test {DIM}({robot_ip}){RST}\n")

    # Don't run if server is active — self-tests lock joints mid-control
    pid_path = pid_file_for_ip(robot_ip)
    if os.path.exists(pid_path):
        try:
            with open(pid_path) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            print(f"  {RED}Error:{RST} Server is running (PID {pid}).")
            print(f"  Stop it first with: {BOLD}aiofranka stop{RST}\n")
            return
        except (ProcessLookupError, ValueError):
            pass

    total = 4
    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)

        # Step 1: Check status
        st = _cli_run_with_spinner("Checking self-test status", 1, total,
                                   client.get_self_test_status)
        status = st.get("status", "unknown")
        remaining = st.get("remaining", 0)

        if status == "Running":
            print(f"\n  Self-tests are already running. Wait for them to finish.\n")
            return

        if status == "OK":
            hours_left = remaining / 3600
            print(f"\n  Self-tests are {GREEN}not due{RST} yet ({hours_left:.1f}h remaining).")
            if not args.force:
                print(f"  Use {BOLD}--force{RST} to run them anyway.\n")
                return
            print(f"  Running anyway (--force).\n")
        elif status == "Warning":
            hours_left = remaining / 3600
            print(f"\n  Self-tests are {YELLOW}due soon{RST} ({hours_left:.1f}h remaining).\n")
        elif status == "Elapsed":
            print(f"\n  Self-tests are {RED}overdue{RST}.\n")

        # Step 2: Acquire token (reuse saved token if valid)
        saved_token, saved_token_id = _load_token_state(robot_ip)
        if saved_token is not None:
            client._token = saved_token
            client._token_id = saved_token_id
            if not client.validate_token():
                # Saved token is stale — release it so take_token won't deadlock
                try:
                    client.release_token(best_effort=True)
                except Exception:
                    pass
                _clear_token(robot_ip)
                client._token = None
                client._token_id = None

        if client._token is None:
            _cli_run_with_spinner("Acquiring control token", 2, total,
                                  client.take_token, timeout=15)
        else:
            print(_cli_step_line(2, total, "Acquiring control token",
                                 f"{GREEN}done{RST} {DIM}(reused){RST}"))

        try:
            # Step 3: Run self-tests
            _cli_run_with_spinner("Running self-tests", 3, total,
                                  client.execute_self_tests)

            # Step 4: Check new status
            st2 = _cli_run_with_spinner("Checking results", 4, total,
                                         client.get_self_test_status)
            new_remaining = st2.get("remaining", 0)
            if new_remaining:
                hours_left = new_remaining / 3600
                print(f"\n  {GREEN}Self-tests passed.{RST} Next due in {hours_left:.1f}h.\n")
            else:
                print(f"\n  {GREEN}Self-tests passed.{RST}\n")
        finally:
            client.release_token(best_effort=True)

    except Exception as e:
        sys.stdout.write(f"\r{' ' * 70}\r")
        print(f"  {RED}Error:{RST} {e}\n")


# ── Log viewer ─────────────────────────────────────────────────────────────

LOG_PATH = os.path.join(CONFIG_DIR, "server.log")


def cmd_log(args):
    """Show recent server log lines, optionally follow."""
    if not os.path.exists(LOG_PATH):
        print(f"  No log file found at {LOG_PATH}")
        return

    n = args.n

    if not args.follow:
        # Just print the last n lines
        with open(LOG_PATH, "rb") as f:
            # Seek from end to find last n newlines
            try:
                f.seek(0, 2)
                fsize = f.tell()
            except OSError:
                f.seek(0)
                lines = f.readlines()
                for line in lines[-n:]:
                    sys.stdout.write(line.decode(errors="replace"))
                return

            pos = fsize
            newlines = 0
            while pos > 0 and newlines < n + 1:
                pos = max(pos - 4096, 0)
                f.seek(pos)
                newlines = f.read(fsize - pos).count(b"\n")

            f.seek(pos)
            lines = f.readlines()
            for line in lines[-n:]:
                sys.stdout.write(line.decode(errors="replace"))
    else:
        # tail -f behavior
        import subprocess
        try:
            subprocess.run(["tail", "-n", str(n), "-f", LOG_PATH])
        except KeyboardInterrupt:
            pass


# libfranka's docs budget roughly 300us for the user's own control loop. The
# rest of the 1 ms cycle goes to the network round trip and the robot's own
# processing; a command that misses the cycle is dropped by the robot.
RESPONSE_BUDGET_US = 300.0

# A dropped command shows up in the success rate of the next one or two states,
# so look this many iterations back for its cause.
DROP_LOOKBACK = 3


def _analyze_fci_timing(phase_all, success_rate_all, robot_time_all):
    """Compute the timing metrics that matter on the robot's side of the loop.

    - response: time from readOnce() returning to writeOnce() returning
      (mj_fwd + state_build + ctrl_law), i.e. how long the robot waits for the
      command after its state reached us.
    - skipped states: gaps in robot_state.time larger than 1 ms. The loop never
      saw these states and sent no command for them.
    - dropped commands: control_command_success_rate is the fraction of the last
      100 commands the robot received in time. Each dropped command lowers it
      by 0.01 for 100 ticks, so the sum of (1 - rate) is the number of dropped
      commands. Drop events are the iterations where the rate decreases.
    """
    import numpy as np

    response = phase_all[:, 1] + phase_all[:, 2] + phase_all[:, 3]

    robot_ms = np.rint(robot_time_all * 1000.0).astype(np.int64)
    skipped = np.zeros(len(robot_ms), dtype=np.int64)
    skipped[1:] = np.maximum(np.diff(robot_ms) - 1, 0)

    sr = success_rate_all
    new_drops = np.zeros(len(sr), dtype=np.int64)
    new_drops[1:] = np.maximum(np.rint((sr[:-1] - sr[1:]) * 100), 0)
    drop_events = np.flatnonzero(new_drops)

    causes = {"slow response": 0, "skipped state": 0, "network/wake-up": 0}
    for i in drop_events:
        lo = max(0, i - DROP_LOOKBACK)
        if np.any(response[lo:i] > RESPONSE_BUDGET_US):
            causes["slow response"] += 1
        elif np.any(skipped[lo:i + 1]):
            causes["skipped state"] += 1
        else:
            causes["network/wake-up"] += 1

    return {
        "response": response,
        "resp_over": int(np.sum(response > RESPONSE_BUDGET_US)),
        "skipped": int(np.sum(skipped)),
        "skip_gaps": int(np.count_nonzero(skipped)),
        "max_gap_ms": int(np.max(skipped)) + 1 if np.any(skipped) else 1,
        "dropped": int(round(np.sum(1.0 - sr))),
        "drop_events": len(drop_events),
        "drop_causes": causes,
    }


def _run_bench_loop(robot, duration, cpu_pin=None, sched_fifo=None,
                    disable_gc=False, mlock=False, prealloc=False):
    """Run the core benchmark loop.

    Returns (dt_array, phase_array, phase_names, success_rate_array,
    robot_time_array), where robot_time_array holds robot_state.time in seconds.

    Applies RT tuning (cpu pin, SCHED_FIFO, gc, mlock, prealloc) during the
    measured window only.
    """
    import gc
    import mujoco
    import numpy as np
    import pylibfranka
    from aiofranka.robot import set_macos_control_thread_qos

    model = robot.model
    data = robot.data
    site_id = robot.site_id
    tc = robot.torque_controller

    PHASES = ["readOnce", "mj_fwd", "state_build", "ctrl_law", "shm_write"]
    n_iters = int(duration * 1000)
    dt_all = np.empty(n_iters, dtype=np.float64)
    phase_all = np.empty((n_iters, len(PHASES)), dtype=np.float64)
    success_rate_all = np.empty(n_iters, dtype=np.float64)
    robot_time_all = np.empty(n_iters, dtype=np.float64)

    # Pre-allocate reusable buffers (used when prealloc=True)
    _ee = np.eye(4)
    _jac = np.zeros((6, 7))
    _mm = np.zeros((7, 7))
    _q = np.empty(7)
    _dq = np.empty(7)
    _kp = np.ones(7) * 80
    _kd = np.ones(7) * 4
    _tau = np.empty(7)
    _state_qpos = np.empty(7)
    _state_qvel = np.empty(7)
    _state_ctrl = np.empty(7)

    # Warm up (500 iterations = 0.5s)
    for _ in range(500):
        robot_state, _ = tc.readOnce()
        data.qpos[:] = robot_state.q
        data.qvel[:] = robot_state.dq
        data.ctrl[:] = robot_state.tau_J_d
        mujoco.mj_forward(model, data)
        torque_cmd = pylibfranka.Torques([0.0] * 7)
        torque_cmd.motion_finished = False
        tc.writeOnce(torque_cmd)

    # Apply RT tuning
    old_affinity = None
    applied_fifo = False
    gc_was_enabled = gc.isenabled()
    if sys.platform == "darwin":
        set_macos_control_thread_qos()
    if cpu_pin is not None and hasattr(os, "sched_setaffinity"):
        old_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu_pin})
    if sched_fifo is not None and hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(sched_fifo))
            applied_fifo = True
        except PermissionError:
            pass
    if disable_gc:
        gc.collect()  # collect now, then disable
        gc.disable()
    if mlock:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            MCL_CURRENT = 1
            MCL_FUTURE = 2
            libc.mlockall(MCL_CURRENT | MCL_FUTURE)
        except Exception:
            pass

    if prealloc:
        last_t = time.perf_counter()
        for i in range(n_iters):
            t0 = time.perf_counter()
            robot_state, _ = tc.readOnce()
            t1 = time.perf_counter()

            data.qpos[:] = robot_state.q
            data.qvel[:] = robot_state.dq
            data.ctrl[:] = robot_state.tau_J_d
            mujoco.mj_forward(model, data)
            t2 = time.perf_counter()

            ee_xyz = data.site(site_id).xpos
            ee_mat = data.site(site_id).xmat.reshape(3, 3)
            _ee[:3, :3] = ee_mat
            _ee[:3, 3] = ee_xyz
            _jac[:] = 0
            mujoco.mj_jacSite(model, data, _jac[:3], _jac[3:], site_id)
            _mm[:] = 0
            mujoco.mj_fullM(model, data, _mm)
            t3 = time.perf_counter()

            np.copyto(_q, data.qpos)
            np.copyto(_dq, data.qvel)
            np.subtract(_q, _q, out=_tau)
            np.multiply(_kp, _tau, out=_tau)
            np.subtract(_tau, _kd * _dq, out=_tau)
            torque_cmd = pylibfranka.Torques(_tau.tolist())
            torque_cmd.motion_finished = False
            tc.writeOnce(torque_cmd)
            t4 = time.perf_counter()

            np.copyto(_state_qpos, data.qpos)
            np.copyto(_state_qvel, data.qvel)
            np.copyto(_state_ctrl, data.ctrl)
            t5 = time.perf_counter()

            dt_all[i] = (t0 - last_t) * 1e6
            phase_all[i, 0] = (t1 - t0) * 1e6
            phase_all[i, 1] = (t2 - t1) * 1e6
            phase_all[i, 2] = (t3 - t2) * 1e6
            phase_all[i, 3] = (t4 - t3) * 1e6
            phase_all[i, 4] = (t5 - t4) * 1e6
            success_rate_all[i] = robot_state.control_command_success_rate
            robot_time_all[i] = robot_state.time.to_sec()
            last_t = t0
    else:
        last_t = time.perf_counter()
        for i in range(n_iters):
            t0 = time.perf_counter()
            robot_state, _ = tc.readOnce()
            t1 = time.perf_counter()

            data.qpos[:] = robot_state.q
            data.qvel[:] = robot_state.dq
            data.ctrl[:] = robot_state.tau_J_d
            mujoco.mj_forward(model, data)
            t2 = time.perf_counter()

            ee_xyz = data.site(site_id).xpos
            ee_mat = data.site(site_id).xmat.reshape(3, 3)
            ee = np.eye(4)
            ee[:3, :3] = ee_mat
            ee[:3, 3] = ee_xyz
            jac = np.zeros((6, 7))
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            mm = np.zeros((7, 7))
            mujoco.mj_fullM(model, data, mm)
            t3 = time.perf_counter()

            q = np.array(data.qpos)
            dq = np.array(data.qvel)
            kp = np.ones(7) * 80
            kd = np.ones(7) * 4
            tau = kp * (q - q) - kd * dq
            torque_cmd = pylibfranka.Torques(tau.tolist())
            torque_cmd.motion_finished = False
            tc.writeOnce(torque_cmd)
            t4 = time.perf_counter()

            _ = {
                "qpos": np.array(data.qpos), "qvel": np.array(data.qvel),
                "ee": ee.copy(), "jac": jac.copy(), "mm": mm.copy(),
                "last_torque": np.array(data.ctrl),
            }
            t5 = time.perf_counter()

            dt_all[i] = (t0 - last_t) * 1e6
            phase_all[i, 0] = (t1 - t0) * 1e6
            phase_all[i, 1] = (t2 - t1) * 1e6
            phase_all[i, 2] = (t3 - t2) * 1e6
            phase_all[i, 3] = (t4 - t3) * 1e6
            phase_all[i, 4] = (t5 - t4) * 1e6
            success_rate_all[i] = robot_state.control_command_success_rate
            robot_time_all[i] = robot_state.time.to_sec()
            last_t = t0

    # Restore
    if disable_gc and gc_was_enabled:
        gc.enable()
    if applied_fifo:
        try:
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        except Exception:
            pass
    if old_affinity is not None:
        os.sched_setaffinity(0, old_affinity)
    if mlock:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.munlockall()
        except Exception:
            pass

    return dt_all, phase_all, PHASES, success_rate_all, robot_time_all


def _run_all_combos(args):
    """Run benchmark with all RT setting combinations and print comparison."""
    import numpy as np
    from aiofranka.robot import RobotInterface
    from aiofranka.server import (
        _DeskClientV2, _load_token_state, _save_token_state, _clear_token,
    )

    robot_ip = _resolve_ip(args.ip)
    username, password = _resolve_credentials(args)
    protocol = args.protocol
    duration = args.duration

    print(f"\n  {BOLD}aiofranka rt-benchmark{RST} {DIM}|{RST} all combos {DIM}({robot_ip}){RST}")
    print(f"  {DIM}Duration: {duration:.0f}s per run{RST}\n")

    # --- Setup ---
    setup_total = 4
    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
        saved_token, saved_token_id = _load_token_state(robot_ip)
        if saved_token is not None:
            client._token = saved_token
            client._token_id = saved_token_id
            if not client.validate_token():
                client._token = None
                client._token_id = None
        if client._token is None:
            _cli_run_with_spinner("Acquiring control token", 1, setup_total,
                                  client.take_token, timeout=15)
        else:
            print(_cli_step_line(1, setup_total, "Acquiring control token",
                                 f"{GREEN}done{RST} {DIM}(reused){RST}"))
        _cli_run_with_spinner("Recovering safety errors", 2, setup_total,
                              client.recover_errors)
        if client.are_joints_unlocked():
            print(_cli_step_line(3, setup_total, "Unlocking joints",
                                 f"{GREEN}done{RST} {DIM}(already){RST}"))
        else:
            _cli_run_with_spinner("Unlocking joints", 3, setup_total,
                                  client.unlock)
        if client.is_fci_active():
            print(_cli_step_line(4, setup_total, "Activating FCI",
                                 f"{GREEN}done{RST} {DIM}(already){RST}"))
        else:
            _cli_run_with_spinner("Activating FCI", 4, setup_total,
                                  client.activate_fci)
        _save_token_state(robot_ip, client._token, client._token_id)
    except Exception:
        try:
            client.release_token(best_effort=True)
            _clear_token(robot_ip)
        except Exception:
            pass
        raise

    robot = RobotInterface(robot_ip, read_tool=False)
    robot.start()

    # Use last P-core (i9-14900K: cores 0-15 are P-cores, 16-31 are E-cores)
    n_cpus = os.cpu_count() or 1
    # Pick last core (often least busy) and a mid-range core
    last_core = n_cpus - 1
    # (label, cpu_pin, sched_fifo, disable_gc, mlock, prealloc)
    C = last_core
    combos = [
        ("baseline",              None, None, False, False, False),
        (f"cpu={C}+FIFO",        C,    80,   False, False, False),
        (f"cpu={C}+FIFO+nogc",   C,    80,   True,  False, False),
        (f"cpu={C}+FIFO+mlock",  C,    80,   False, True,  False),
        (f"cpu={C}+FIFO+prealloc", C,  80,   False, False, True),
        (f"cpu={C}+FIFO+all",    C,    80,   True,  True,  True),
    ]
    if sys.platform == "darwin":
        # No CPU pinning, SCHED_FIFO or mlockall on macOS
        combos = [
            ("baseline",              None, None, False, False, False),
            ("nogc",                  None, None, True,  False, False),
            ("prealloc",              None, None, False, False, True),
            ("nogc+prealloc",         None, None, True,  False, True),
        ]

    results = []
    try:
        for label, cpu_pin, sched_fifo, dis_gc, ml, prealloc in combos:
            sys.stdout.write(f"  Running: {BOLD}{label}{RST} ...")
            sys.stdout.flush()
            dt_all, phase_all, phases, sr_all, rt_all = _run_bench_loop(
                robot, duration, cpu_pin=cpu_pin, sched_fifo=sched_fifo,
                disable_gc=dis_gc, mlock=ml, prealloc=prealloc,
            )
            dt = dt_all[1:]
            in_spec = np.sum((dt >= 900) & (dt <= 1100))
            pct_in = in_spec * 100.0 / len(dt)
            sr_min = np.min(sr_all)
            fci = _analyze_fci_timing(phase_all, sr_all, rt_all)
            resp_p999 = np.percentile(fci["response"], 99.9)
            results.append({
                "label": label,
                "pct_in": pct_in,
                "mean": np.mean(dt),
                "std": np.std(dt),
                "max": np.max(dt),
                "p99": np.percentile(dt, 99),
                "p999": np.percentile(dt, 99.9),
                "sr_min": sr_min,
                "resp_p999": resp_p999,
                "dropped": fci["dropped"],
                "skipped": fci["skipped"],
            })
            color = GREEN if pct_in >= 99 else YELLOW if pct_in >= 95 else RED
            sr_color = GREEN if sr_min >= 0.99 else YELLOW if sr_min >= 0.95 else RED
            sys.stdout.write(f"\r  {BOLD}{label:<25}{RST} {color}{pct_in:.2f}%{RST} in spec, "
                             f"std={np.std(dt):.1f}us, p99={np.percentile(dt, 99):.0f}us, "
                             f"max={np.max(dt):.0f}us, "
                             f"sr_min={sr_color}{sr_min:.4f}{RST}, "
                             f"resp_p99.9={resp_p999:.0f}us, dropped={fci['dropped']}, "
                             f"skipped={fci['skipped']}\n")
            sys.stdout.flush()
    finally:
        robot.stop()
        try:
            client.release_token(best_effort=True)
            _clear_token(robot_ip)
        except Exception:
            pass

    # --- Comparison table ---
    print(f"\n  {BOLD}=== Comparison ==={RST}\n")
    print(f"    {'Config':<25} {'In-spec':>8} {'std':>8} {'p99':>8} {'p99.9':>8} {'max':>8} {'sr_min':>8}"
          f" {'resp99.9':>9} {'dropped':>8} {'skipped':>8}")
    print(f"    {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")
    best = max(results, key=lambda r: r["pct_in"])
    for r in results:
        is_best = r is best
        marker = f" {GREEN}★{RST}" if is_best else ""
        color = GREEN if r["pct_in"] >= 99 else YELLOW if r["pct_in"] >= 95 else RED
        sr_color = GREEN if r["sr_min"] >= 0.99 else YELLOW if r["sr_min"] >= 0.95 else RED
        print(f"    {r['label']:<25} {color}{r['pct_in']:>7.2f}%{RST} "
              f"{r['std']:>7.1f} {r['p99']:>7.0f} {r['p999']:>7.0f} {r['max']:>7.0f} "
              f"{sr_color}{r['sr_min']:>7.4f}{RST} {r['resp_p999']:>9.0f} "
              f"{r['dropped']:>8} {r['skipped']:>8}{marker}")
    print(f"\n  {GREEN}Best: {best['label']}{RST}\n")


def cmd_gripper(args):
    """Open or close the Robotiq gripper."""
    from aiofranka.gripper_remote import GripperRemoteController

    gripper = GripperRemoteController(args.port)
    gripper.start()
    gripper.speed = args.speed
    gripper.force = args.force

    if args.open:
        print("Opening gripper...")
        gripper.open()
    else:
        print("Closing gripper...")
        gripper.close()

    gripper.wait_until_reached(timeout=5.0)
    gripper.stop()


def cmd_rt_benchmark(args):
    """Benchmark the 1kHz control loop real-time performance."""
    import mujoco
    import numpy as np
    from aiofranka.robot import RobotInterface
    from aiofranka.server import (
        _DeskClientV2, _load_token_state, _save_token_state, _clear_token,
    )

    robot_ip = _resolve_ip(args.ip)
    username, password = _resolve_credentials(args)
    protocol = args.protocol
    duration = args.duration
    use_v2 = args.v2

    cpu_pin = args.cpu_pin
    sched_fifo = args.sched_fifo
    all_combos = args.all_combos

    if all_combos:
        _run_all_combos(args)
        return

    if sys.platform == "darwin" and (cpu_pin is not None or sched_fifo is not None):
        print(f"\n  {DIM}--cpu-pin and --sched-fifo are not available on macOS, ignoring them{RST}")
        cpu_pin = sched_fifo = None

    mode_label = "v2 (RT thread)" if use_v2 else "v1 (asyncio)"
    rt_flags = []
    if cpu_pin is not None:
        rt_flags.append(f"cpu={cpu_pin}")
    if sched_fifo is not None:
        rt_flags.append(f"FIFO={sched_fifo}")
    rt_label = f" [{', '.join(rt_flags)}]" if rt_flags else ""
    print(f"\n  {BOLD}aiofranka rt-benchmark{RST} {DIM}|{RST} {mode_label}{rt_label} {DIM}({robot_ip}){RST}")
    print(f"  {DIM}Duration: {duration:.0f}s — hold position with zero torque{RST}\n")

    # --- Setup: unlock + FCI ---
    setup_total = 4
    try:
        client = _DeskClientV2(robot_ip, username, password, protocol=protocol)
        saved_token, saved_token_id = _load_token_state(robot_ip)
        if saved_token is not None:
            client._token = saved_token
            client._token_id = saved_token_id
            if not client.validate_token():
                client._token = None
                client._token_id = None

        if client._token is None:
            _cli_run_with_spinner("Acquiring control token", 1, setup_total,
                                  client.take_token, timeout=15)
        else:
            print(_cli_step_line(1, setup_total, "Acquiring control token",
                                 f"{GREEN}done{RST} {DIM}(reused){RST}"))

        _cli_run_with_spinner("Recovering safety errors", 2, setup_total,
                              client.recover_errors)

        if client.are_joints_unlocked():
            print(_cli_step_line(3, setup_total, "Unlocking joints",
                                 f"{GREEN}done{RST} {DIM}(already){RST}"))
        else:
            _cli_run_with_spinner("Unlocking joints", 3, setup_total,
                                  client.unlock)

        if client.is_fci_active():
            print(_cli_step_line(4, setup_total, "Activating FCI",
                                 f"{GREEN}done{RST} {DIM}(already){RST}"))
        else:
            _cli_run_with_spinner("Activating FCI", 4, setup_total,
                                  client.activate_fci)

        _save_token_state(robot_ip, client._token, client._token_id)

    except Exception:
        try:
            client.release_token(best_effort=True)
            _clear_token(robot_ip)
        except Exception:
            pass
        raise

    # --- Run benchmark ---
    print(f"\n  {YELLOW}Running benchmark...{RST}\n")

    robot = RobotInterface(robot_ip, read_tool=False)
    robot.start()

    PHASES = ["readOnce", "mj_fwd", "state_build", "ctrl_law", "shm_write"]

    try:
        dt_all, phase_all, PHASES, success_rate, robot_time = _run_bench_loop(
            robot, duration, cpu_pin=cpu_pin, sched_fifo=sched_fifo
        )
    finally:
        robot.stop()
        try:
            client.release_token(best_effort=True)
            _clear_token(robot_ip)
        except Exception:
            pass

    # --- Print report ---
    # dt_all[i] = t0[i] - t0[i-1] = loop time of iteration i-1
    # phase_all[i] = phases of iteration i
    # To align: dt[k] = dt_all[k+1] = loop time of iteration k
    #           phases[k] = phase_all[k] = phases of iteration k
    dt = dt_all[1:]       # skip first (no previous t0)
    phases = phase_all[:-1]  # drop last (no dt for it)
    total_compute = phases.sum(axis=1)

    print(f"  {BOLD}=== RT Benchmark Results ==={RST}")
    print(f"  {DIM}Samples: {len(dt)}, Duration: {duration:.0f}s, Mode: {mode_label}{rt_label}{RST}\n")

    # Iteration timing
    print(f"  {BOLD}Iteration timing (target: 1000us){RST}")
    print(f"    mean   = {np.mean(dt):.1f} us")
    print(f"    std    = {np.std(dt):.1f} us")
    print(f"    min    = {np.min(dt):.1f} us")
    print(f"    max    = {np.max(dt):.1f} us")
    print(f"    jitter = {np.max(dt) - np.min(dt):.1f} us")
    print()

    # Percentiles
    print(f"  {BOLD}Percentiles (us){RST}")
    for p in [50, 90, 95, 99, 99.9]:
        val = np.percentile(dt, p)
        marker = f"  {RED}<--{RST}" if val > 1100 else ""
        print(f"    p{p:<5} = {val:.1f}{marker}")
    print()

    # Jitter budget (how many iterations outside 0.9-1.1ms)
    in_spec = np.sum((dt >= 900) & (dt <= 1100))
    out_spec = len(dt) - in_spec
    pct_in = in_spec * 100.0 / len(dt)
    color = GREEN if pct_in >= 99 else YELLOW if pct_in >= 95 else RED
    print(f"  {BOLD}Timing accuracy{RST}")
    print(f"    In spec (900-1100us):  {color}{pct_in:.2f}%{RST} ({in_spec}/{len(dt)})")
    print(f"    Out of spec:           {out_spec}")
    print()

    # Control command success rate
    sr = success_rate[:-1]  # align with dt (drop last, same as phases)
    sr_mean = np.mean(sr)
    sr_min = np.min(sr)
    sr_color = GREEN if sr_min >= 0.99 else YELLOW if sr_min >= 0.95 else RED
    print(f"  {BOLD}Control command success rate{RST}")
    print(f"    mean   = {sr_color}{sr_mean:.4f}{RST}")
    print(f"    min    = {sr_color}{sr_min:.4f}{RST}")
    n_below = np.sum(sr < 1.0)
    if n_below > 0:
        print(f"    below 1.0 = {RED}{n_below}{RST} iterations")
    else:
        print(f"    below 1.0 = {GREEN}0{RST} iterations")
    print()

    # What the robot sees: response time, skipped states and dropped commands
    fci = _analyze_fci_timing(phase_all, success_rate, robot_time)
    resp = fci["response"]
    n_ticks = len(resp)

    resp_pct = fci["resp_over"] * 100.0 / n_ticks
    resp_color = GREEN if fci["resp_over"] == 0 else YELLOW if resp_pct < 0.1 else RED
    print(f"  {BOLD}Response time (readOnce -> writeOnce, budget {RESPONSE_BUDGET_US:.0f}us){RST}")
    print(f"    mean   = {np.mean(resp):.1f} us")
    print(f"    p99    = {np.percentile(resp, 99):.1f} us")
    print(f"    p99.9  = {np.percentile(resp, 99.9):.1f} us")
    print(f"    max    = {np.max(resp):.1f} us")
    print(f"    over   = {resp_color}{fci['resp_over']}{RST} iterations ({resp_pct:.2f}%)")
    print()

    print(f"  {BOLD}Robot clock (robot_state.time){RST}")
    if fci["skipped"] > 0:
        print(f"    skipped = {RED}{fci['skipped']}{RST} states in {fci['skip_gaps']} gaps "
              f"(max gap {fci['max_gap_ms']} ms)")
    else:
        print(f"    skipped = {GREEN}0{RST} states (every tick seen)")
    print()

    drop_pct = fci["dropped"] * 100.0 / n_ticks
    drop_color = GREEN if fci["dropped"] == 0 else YELLOW if drop_pct < 1.0 else RED
    print(f"  {BOLD}Dropped commands (robot reused the previous torque){RST}")
    print(f"    dropped \u2248 {drop_color}{fci['dropped']}{RST} ({drop_pct:.2f}% of ticks) "
          f"in {fci['drop_events']} events")
    if fci["drop_events"] > 0:
        print(f"    {DIM}Likely cause, within {DROP_LOOKBACK} iterations before each event:{RST}")
        for cause, count in fci["drop_causes"].items():
            pct = count * 100.0 / fci["drop_events"]
            print(f"      {cause:<16} {count:>5}  ({pct:5.1f}%)")
    print()

    # Per-phase breakdown
    print(f"  {BOLD}Per-phase breakdown (us){RST}")
    print(f"    {'Phase':<14} {'mean':>8} {'std':>8} {'p99':>8} {'p99.9':>8} {'max':>8}")
    print(f"    {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for j, name in enumerate(PHASES):
        col = phases[:, j]
        print(f"    {name:<14} {np.mean(col):>8.1f} {np.std(col):>8.1f} "
              f"{np.percentile(col, 99):>8.1f} {np.percentile(col, 99.9):>8.1f} "
              f"{np.max(col):>8.1f}")
    print(f"    {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"    {'TOTAL':<14} {np.mean(total_compute):>8.1f} {np.std(total_compute):>8.1f} "
          f"{np.percentile(total_compute, 99):>8.1f} {np.percentile(total_compute, 99.9):>8.1f} "
          f"{np.max(total_compute):>8.1f}")
    print()

    # Jitter attribution: for out-of-spec iterations, which phase spiked?
    out_mask = (dt < 900) | (dt > 1100)
    n_out = np.sum(out_mask)
    if n_out > 0:
        print(f"  {BOLD}Jitter attribution ({n_out} out-of-spec iterations){RST}")
        # For each phase, compute its median value across ALL iterations
        phase_medians = np.median(phases, axis=0)
        # For out-of-spec iterations, compute excess over median for each phase
        out_phases = phases[out_mask]
        excess = out_phases - phase_medians  # excess time vs typical
        # Count how often each phase was the dominant contributor
        dominant = np.argmax(excess, axis=1)
        print(f"    {'Phase':<14} {'blamed':>7} {'%blamed':>8} {'avg_excess':>11} {'max_excess':>11}")
        print(f"    {'─'*14} {'─'*7} {'─'*8} {'─'*11} {'─'*11}")
        for j, name in enumerate(PHASES):
            mask_j = dominant == j
            count_j = np.sum(mask_j)
            pct_j = count_j * 100.0 / n_out if n_out > 0 else 0
            avg_exc = np.mean(excess[mask_j, j]) if count_j > 0 else 0
            max_exc = np.max(excess[mask_j, j]) if count_j > 0 else 0
            bar_len = int(pct_j * 20 / 100) if pct_j > 0 else 0
            bar = "█" * bar_len
            color_code = RED if pct_j > 40 else YELLOW if pct_j > 15 else DIM
            print(f"    {name:<14} {count_j:>7} {color_code}{pct_j:>7.1f}%{RST} "
                  f"{avg_exc:>10.1f}us {max_exc:>10.1f}us  {color_code}{bar}{RST}")
        print()

        # Show the top 10 worst iterations with per-phase detail
        worst_idx = np.argsort(dt)[-10:][::-1]
        print(f"  {BOLD}Top 10 worst iterations{RST}")
        header_phases = "  ".join(f"{name:>8}" for name in PHASES)
        print(f"    {'#':>6} {'dt':>8}  {header_phases}  {'blame':>10}")
        print(f"    {'─'*6} {'─'*8}  {'─' * (10 * len(PHASES) - 2)}  {'─'*10}")
        for idx in worst_idx:
            phase_vals = "  ".join(f"{phases[idx, j]:>8.1f}" for j in range(len(PHASES)))
            blame_j = np.argmax(phases[idx] - phase_medians)
            blame_name = PHASES[blame_j]
            print(f"    {idx:>6} {dt[idx]:>8.1f}  {phase_vals}  {RED}{blame_name:>10}{RST}")
        print()

    # ASCII histogram of iteration times
    print(f"  {BOLD}Iteration time distribution (us){RST}")
    bin_edges = [0, 800, 900, 950, 1000, 1050, 1100, 1200, 1500, 2000, float('inf')]
    bin_labels = ["<800", "800-900", "900-950", "950-1000", "1000-1050",
                  "1050-1100", "1100-1200", "1200-1500", "1500-2000", ">2000"]
    counts = np.histogram(dt, bins=bin_edges)[0]
    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 30
    for label, count in zip(bin_labels, counts):
        bar_len = int(count * bar_width / max_count)
        bar = "\u2588" * bar_len
        pct = count * 100.0 / len(dt)
        if pct > 0.01:
            color_code = GREEN if "950" in label or "1000" in label or "1050" in label else (
                YELLOW if "900" in label or "1100" in label else RED
            )
            print(f"    {label:>10} | {color_code}{bar:<{bar_width}}{RST} {count:>6} ({pct:.1f}%)")
    print()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="aiofranka", description="aiofranka control server")
    subparsers = parser.add_subparsers(dest="command")

    # start-server
    p_start = subparsers.add_parser("start-server", help="Start the control server")
    p_start.add_argument("--ip", type=str, default=None, help="Robot IP (default: last used or 172.16.0.2)")
    p_start.add_argument("--foreground", action="store_true", help="Run in foreground (don't daemonize)")
    p_start.add_argument("--no-unlock", action="store_true", help="Skip auto unlock/FCI activation")
    p_start.add_argument("--username", type=str, default="admin", help="Robot web UI username (prompts if not saved)")
    p_start.add_argument("--password", type=str, default="admin", help="Robot web UI password (prompts if not saved)")
    p_start.add_argument("--protocol", type=str, default="https", choices=["http", "https"],
                         help="Robot web UI protocol (default: https)")
    p_start.add_argument("--lock-on-error", action="store_true",
                         help="Lock joints when the server dies due to a control error (default: leave unlocked)")
    p_start.add_argument("--no-home", action="store_true",
                         help="Skip moving to home pose on startup")
    p_start.add_argument("--native", action="store_true",
                         help="Run the 1 kHz control loop in C++ (NativeServerController)")

    # gravcomp
    p_gravcomp = subparsers.add_parser("gravcomp", help="Gravity compensation mode (foreground, Ctrl+C to stop)")
    p_gravcomp.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_gravcomp.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_gravcomp.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_gravcomp.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
    p_gravcomp.add_argument("--mode", type=str, default="execute", choices=["execute", "program"],
                            help="execute: gravity compensation over FCI, in the foreground (default); "
                                 "program: switch to Programming mode, where the arm moves while the "
                                 "guiding button on the end effector is held")
    p_gravcomp.add_argument("--damping", type=float, default=0.0,
                            help="Joint velocity damping (kd) per joint, in execute mode (default: 0)")
    p_gravcomp.add_argument("--http-port", type=int, default=0,
                            help="Serve GET /qpos on this port (e.g. 8080), in execute mode")

    # home
    p_home = subparsers.add_parser("home", help="Move robot to home position")
    p_home.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_home.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_home.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_home.add_argument("--protocol", type=str, default="https", choices=["http", "https"])

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop the control server")
    p_stop.add_argument("--ip", type=str, default=None, help="Robot IP")

    # status
    p_status = subparsers.add_parser("status", help="Check server status")
    p_status.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_status.add_argument("--protocol", type=str, default="https", choices=["http", "https"])

    # lock
    p_lock = subparsers.add_parser("lock", help="Lock robot joints (close brakes)")
    p_lock.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_lock.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_lock.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_lock.add_argument("--protocol", type=str, default="https", choices=["http", "https"])

    # unlock
    p_unlock = subparsers.add_parser("unlock", help="Unlock robot joints (open brakes)")
    p_unlock.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_unlock.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_unlock.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_unlock.add_argument("--protocol", type=str, default="https", choices=["http", "https"])

    # selftest
    p_selftest = subparsers.add_parser("selftest", help="Run safety self-tests")
    p_selftest.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_selftest.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_selftest.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_selftest.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
    p_selftest.add_argument("--force", action="store_true", help="Run even if not due yet")

    # mode
    p_mode = subparsers.add_parser("mode", help="View or change the operating mode")
    p_mode.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_mode.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_mode.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_mode.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
    p_mode.add_argument("mode", nargs="?", choices=["program", "execute"],
                        help="switch to Programming (hand guiding, with the brakes open) or "
                             "Execution (FCI); without it, show the current mode")

    # config
    p_config = subparsers.add_parser("config", help="View/set end-effector configuration")
    p_config.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_config.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_config.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_config.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
    p_config.add_argument("--mass", type=float, default=None, help="End-effector mass in kg")
    p_config.add_argument("--com", type=str, default=None,
                          help="Center of mass x,y,z in meters (e.g. 0,-0.01,0.03)")
    p_config.add_argument("--inertia", type=str, default=None,
                          help="Inertia upper triangle x11,x12,x13,x22,x23,x33 in kg*m^2")
    p_config.add_argument("--translation", type=str, default=None,
                          help="Flange-to-EE translation x,y,z in meters")
    p_config.add_argument("--rotation", type=str, default=None,
                          help="Flange-to-EE rotation roll,pitch,yaw in radians")
    p_config.add_argument("--ee-name", type=str, default=None,
                          help="End-effector type (FrankaHand, None, Other, etc.)")

    # tool
    p_tool = subparsers.add_parser("tool", help="Identify and load end-effector profiles in Desk")
    tool_sub = p_tool.add_subparsers(dest="tool_command")

    def tool_parser(name, help):
        p = tool_sub.add_parser(name, help=help)
        p.add_argument("--ip", type=str, default=None, help="Robot IP")
        p.add_argument("--username", type=str, default="admin", help="Robot web UI username")
        p.add_argument("--password", type=str, default="admin", help="Robot web UI password")
        p.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
        return p

    tool_parser("list", "List the end-effector profiles in Desk")
    tool_parser("load", "Activate an end-effector profile").add_argument("name")
    tool_parser("unload", 'Activate the built-in "No End Effector" profile')
    p_tool_remove = tool_parser("remove", "Delete an end-effector profile")
    p_tool_remove.add_argument("name")
    p_tool_remove.add_argument("-y", "--yes", action="store_true", help="Do not ask before deleting")

    p_tool_identify = tool_parser(
        "identify", "Identify the tool on the flange, save it as a profile and activate it (moves the robot)")
    p_tool_identify.add_argument("name")
    p_tool_identify.add_argument("--poses", type=int, default=16, help="Number of poses (default: 16)")
    p_tool_identify.add_argument("--speed", type=float, default=0.5,
                                 help="Peak joint speed in rad/s (default: 0.5)")
    p_tool_identify.add_argument("--tool-length", type=float, default=0.2,
                                 help="Tool length from the flange in m, for collision checks (default: 0.2)")
    p_tool_identify.add_argument("--tool-radius", type=float, default=0.1,
                                 help="Tool radius in m, for collision checks (default: 0.1)")
    p_tool_identify.add_argument("--floor", type=float, default=0.0,
                                 help="Floor or table height in the base frame in m; the robot keeps "
                                      "5 cm from it (default: 0, the mounting plane)")
    p_tool_identify.add_argument("--no-load", action="store_true",
                                 help="Only save the profile, without activating it")
    p_tool_identify.add_argument("-y", "--yes", action="store_true", help="Do not ask before moving or saving")

    # camera
    p_camera = subparsers.add_parser(
        "camera", help="Calibrate a fixed camera against the robot (pip install 'aiofranka[camera]')")
    camera_sub = p_camera.add_subparsers(dest="camera_command")
    p_camera_calibrate = camera_sub.add_parser(
        "calibrate", help="Capture views of the calibration cube while moving the arm by hand in "
                          "Programming mode, and fit")
    p_camera_calibrate.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_camera_calibrate.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_camera_calibrate.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_camera_calibrate.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
    p_camera_calibrate.add_argument("--stream", type=str, default=None,
                                    help="aiocamera stream, e.g. rs_SERIAL_color (default: the only "
                                         "RealSense color stream)")
    p_camera_calibrate.add_argument("--cube", type=str, default=None,
                                    help="AprilCube config.json (default: aprilcube's calibration cube)")
    p_camera_calibrate.add_argument("--out", type=str, default="camera_calibration",
                                    help="Folder for the session folder (default: camera_calibration)")
    p_camera_fit = camera_sub.add_parser("fit", help="Fit a recorded session again")
    p_camera_fit.add_argument("session", help="Session folder with views.json")

    # log
    p_log = subparsers.add_parser("log", help="View server log")
    p_log.add_argument("-n", type=int, default=20, help="Number of lines to show (default: 20)")
    p_log.add_argument("-f", "--follow", action="store_true", help="Follow log output (like tail -f)")

    # gripper
    p_gripper = subparsers.add_parser("gripper", help="Open or close the Robotiq gripper")
    p_gripper_action = p_gripper.add_mutually_exclusive_group(required=True)
    p_gripper_action.add_argument("--open", action="store_true", help="Fully open the gripper")
    p_gripper_action.add_argument("--close", action="store_true", help="Fully close the gripper")
    p_gripper.add_argument("--port", type=str, default="/dev/ttyUSB1", help="Serial port (default: /dev/ttyUSB1)")
    p_gripper.add_argument("--speed", type=int, default=128, help="Gripper speed 1-255 (default: 128)")
    p_gripper.add_argument("--force", type=int, default=128, help="Gripper force 0-255 (default: 128)")

    # rt-benchmark
    p_bench = subparsers.add_parser("rt-benchmark", help="Benchmark 1kHz control loop real-time performance")
    p_bench.add_argument("--ip", type=str, default=None, help="Robot IP")
    p_bench.add_argument("--username", type=str, default="admin", help="Robot web UI username")
    p_bench.add_argument("--password", type=str, default="admin", help="Robot web UI password")
    p_bench.add_argument("--protocol", type=str, default="https", choices=["http", "https"])
    p_bench.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds (default: 10)")
    p_bench.add_argument("--v2", action="store_true", help="Use v2 RT-threaded control loop")
    p_bench.add_argument("--cpu-pin", type=int, default=None, metavar="CORE",
                         help="Pin benchmark thread to specific CPU core (e.g. --cpu-pin 31)")
    p_bench.add_argument("--sched-fifo", type=int, default=None, metavar="PRIO", nargs="?", const=80,
                         help="Set SCHED_FIFO real-time priority (default: 80, requires root/cap)")
    p_bench.add_argument("--all-combos", action="store_true",
                         help="Run all combinations of RT settings and compare")

    args = parser.parse_args()

    if args.command == "start-server":
        cmd_start(args)
    elif args.command == "gravcomp":
        cmd_gravcomp(args)
    elif args.command == "home":
        cmd_home(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "lock":
        cmd_lock(args)
    elif args.command == "unlock":
        cmd_unlock(args)
    elif args.command == "selftest":
        cmd_selftest(args)
    elif args.command == "mode":
        cmd_mode(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "tool":
        if args.tool_command is None:
            p_tool.print_help()
        else:
            cmd_tool(args)
    elif args.command == "camera":
        if args.camera_command is None:
            p_camera.print_help()
        else:
            cmd_camera(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "gripper":
        cmd_gripper(args)
    elif args.command == "rt-benchmark":
        cmd_rt_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
