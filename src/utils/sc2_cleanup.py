import os
import subprocess
import time
from typing import Optional

import psutil
from loguru import logger

_SC2_NAME_MARKERS = ("SC2", "STARCRAFT")


def _is_sc2_process_name(name: Optional[str]) -> bool:
    if not name:
        return False
    upper = name.upper()
    return any(marker in upper for marker in _SC2_NAME_MARKERS)


def count_sc2_processes() -> int:
    """Count live StarCraft II processes (for VRAM / zombie diagnostics)."""
    count = 0
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if _is_sc2_process_name(proc.info.get("name")):
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count


def unregister_sc2_process(process) -> bool:
    """
    Remove ``process`` from burnysc2's ``kill_switch`` list.

    ``SC2Process.__aexit__`` calls ``kill_switch.kill_all()``, which tears down
    every registered instance (including other trainers). Unregister before
    local cleanup so concurrent SC2 sessions survive.
    """
    if process is None:
        return False
    try:
        from sc2.sc2process import kill_switch
    except Exception:
        return False
    to_kill = getattr(kill_switch, "_to_kill", None)
    if not isinstance(to_kill, list):
        return False
    removed = False
    while process in to_kill:
        try:
            to_kill.remove(process)
            removed = True
        except ValueError:
            break
    return removed


def sc2_process_pid(process) -> Optional[int]:
    """Best-effort PID for a burnysc2 ``SC2Process`` (or compatible mock)."""
    if process is None:
        return None
    inner = getattr(process, "_process", None)
    if inner is not None and hasattr(inner, "pid"):
        try:
            return int(inner.pid)
        except (TypeError, ValueError):
            pass
    if hasattr(process, "pid"):
        try:
            return int(process.pid)
        except (TypeError, ValueError):
            pass
    return None


async def close_sc2_process_local(process, *, wait_sec: float = 3.0) -> None:
    """
    Tear down one SC2 process without ``kill_switch.kill_all()``.

    Safe for concurrent build + universal training: only this instance (and its
    process tree) is killed.
    """
    if process is None:
        return
    unregister_sc2_process(process)
    pid = sc2_process_pid(process)
    try:
        close_conn = getattr(process, "_close_connection", None)
        if callable(close_conn):
            await close_conn()
    except Exception as exc:
        logger.debug(f"close_sc2_process_local: connection close failed: {exc}")
    try:
        clean = getattr(process, "_clean", None)
        if callable(clean):
            clean()
    except Exception as exc:
        logger.debug(f"close_sc2_process_local: _clean failed: {exc}")
    kill_sc2_process_tree(pid, wait_sec=wait_sec)


def kill_sc2_process_tree(pid: Optional[int], *, wait_sec: float = 3.0) -> bool:
    """
    Kill one SC2 process and its children, then wait until it is gone.

    Worker-safe: does **not** touch other workers' SC2 instances (unlike
    ``kill_switch_sc2``). Returns True if the PID was known and cleanup ran.
    """
    if pid is None:
        return False
    try:
        root = psutil.Process(int(pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False

    targets = []
    try:
        targets = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        targets = []
    targets.append(root)

    for proc in targets:
        try:
            logger.info(f"Targeted kill of SC2 tree PID: {proc.pid} ({proc.name()})")
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    deadline = time.monotonic() + max(0.0, float(wait_sec))
    while time.monotonic() < deadline:
        alive = False
        for proc in targets:
            try:
                if proc.is_running():
                    alive = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if not alive:
            break
        time.sleep(0.1)
    return True


def kill_switch_sc2(force: bool = False):
    """
    Finds and kills all StarCraft II processes aggressively.
    Can be skipped by setting environment variable SC2_SKIP_KILL_SWITCH=1

    :param force: If True, run taskkill/psutil cleanup even when SC2_SKIP_KILL_SWITCH=1.
                  Used between universal curriculum levels so the next level can bind ports.
                  Opt out of between-level kills with SC2_SKIP_KILL_BETWEEN_LEVELS=1 (see train_universal).
    """
    if not force and os.environ.get("SC2_SKIP_KILL_SWITCH") == "1":
        logger.info("SC2_SKIP_KILL_SWITCH is set. Skipping global process termination.")
        # We still clear the internal list to be safe
        try:
            from sc2.sc2process import SC2Process
            SC2Process.procs = []
        except: pass
        return 0

    if force and os.environ.get("SC2_SKIP_KILL_SWITCH") == "1":
        logger.info("SC2 kill switch (force=True): running cleanup despite SC2_SKIP_KILL_SWITCH.")

    logger.info("Executing SC2 kill switch...")
    
    # 1. Use taskkill on Windows for maximum aggression
    if os.name == 'nt':
        try:
            # Kill by image name, /F is force, /T is tree (kill children)
            subprocess.run(["taskkill", "/F", "/IM", "SC2_x64.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "SC2.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "SC2_x86.exe", "/T"], capture_output=True)
        except Exception as e:
            logger.debug(f"Taskkill failed (probably no processes): {e}")

    # 2. Backup with psutil
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info['name']
            if _is_sc2_process_name(name):
                logger.info(f"Killing SC2 process: {name} (PID: {proc.info['pid']})")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} SC2 processes via psutil.")
    
    # 3. Clear burnysc2's internal tracking to prevent accumulation in logs/logic
    try:
        from sc2.sc2process import SC2Process
        SC2Process.procs = []
        logger.info("Cleared SC2Process.procs list.")
    except Exception as e:
        logger.debug(f"Could not clear SC2Process.procs: {e}")
        
    return killed_count
