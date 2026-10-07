"""
Resolve PyTorch device for SB3 training (sync universal + async learner/collector).

Precedence: env SC2_TRAINING_DEVICE > YAML training.device > auto.

**GPU only:** ``auto``, ``cuda``, and ``gpu`` all resolve to ``cuda:0`` and raise if
CUDA is unavailable. Do not add CPU fallbacks or split collector/learner devices.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from typing import Any, Iterator, Mapping, Optional

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This project requires CUDA for all policy training and inference "
            "(collector, learner, opponent). torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build, check drivers, and CUDA_VISIBLE_DEVICES."
        )
    return torch.device("cuda:0")


def resolve_training_device(training_config: Optional[Mapping[str, Any]] = None) -> torch.device:
    cfg = training_config or {}
    raw = os.environ.get("SC2_TRAINING_DEVICE", "").strip().lower()
    if not raw:
        raw = str(cfg.get("device", "auto")).strip().lower()

    if raw == "cpu":
        raise RuntimeError(
            "training.device / SC2_TRAINING_DEVICE=cpu is not supported. "
            "All SB3 models (collector, learner, opponent) must run on CUDA."
        )

    if raw in ("cuda", "gpu", "auto", "", "default"):
        return _require_cuda()

    raise ValueError(
        f"Invalid training device {raw!r}. Use 'auto' or 'cuda' "
        "(SC2_TRAINING_DEVICE or training.device). CPU is not supported."
    )


def resolve_opponent_device(training_config: Optional[Mapping[str, Any]] = None) -> torch.device:
    """
    Device for opponent inference inside vec-env worker subprocesses.

    Always CUDA (same as collector/learner). Override keys are ignored if they request CPU.
    """
    cfg = dict(training_config or {})
    env_cfg = cfg.get("env") or {}
    raw = os.environ.get("SC2_OPPONENT_DEVICE", "").strip().lower()
    if not raw:
        raw = str(
            cfg.get("opponent_device") or env_cfg.get("opponent_device") or "cuda"
        ).strip().lower()

    if raw == "cpu":
        raise RuntimeError(
            "opponent_device / SC2_OPPONENT_DEVICE=cpu is not supported. "
            "Opponent policies must run on CUDA."
        )

    if raw in ("cuda", "gpu", "auto", "", "default"):
        return resolve_training_device(cfg)

    raise ValueError(
        f"Invalid opponent device {raw!r}. Use 'cuda' or 'auto'. CPU is not supported."
    )


def cuda_init_lock_path() -> str:
    return os.environ.get(
        "SC2_CUDA_INIT_LOCK",
        os.path.join(tempfile.gettempdir(), "sc2_training_cuda_init.lock"),
    )


def opponent_load_lock_path() -> str:
    return os.environ.get(
        "SC2_OPPONENT_LOAD_LOCK",
        os.path.join(tempfile.gettempdir(), "sc2_training_opponent_load.lock"),
    )


@contextlib.contextmanager
def cuda_init_file_lock(timeout: float = 180.0) -> Iterator[None]:
    """
    Cross-process mutex for first CUDA context creation on Windows.

    When the async learner and collector (or rapid spawn retries) call
    ``torch.cuda.init()`` concurrently, some driver stacks crash with
    0xC0000005 / WinError 1114 (shm.dll). Only one process may warm up CUDA at a time.
    """
    with _cross_process_file_lock(cuda_init_lock_path(), timeout=timeout, label="CUDA init"):
        yield


@contextlib.contextmanager
def opponent_load_file_lock(timeout: float = 300.0) -> Iterator[None]:
    """
    Serialize opponent checkpoint loads across SubprocVecEnv workers.

    Each worker holds a full squad policy on the shared GPU; concurrent grafts
    spike VRAM and commonly surface as ``CUDA error: unspecified launch failure``.
    """
    with _cross_process_file_lock(
        opponent_load_lock_path(), timeout=timeout, label="opponent load"
    ):
        yield


def _pid_is_alive(pid: int) -> bool:
    """True when ``pid`` looks like a live OS process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it — treat as alive.
        return True
    except OSError:
        return False
    return True


def _read_lock_owner_pid(lock_path: str) -> Optional[int]:
    try:
        with open(lock_path, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw.split()[0])
    except (TypeError, ValueError):
        return None


# Creator must write+fsync its PID before waiters may treat an empty lock as
# abandoned. Stealing sooner races O_EXCL create vs write and lets two workers
# enter CUDA init together (Windows often surfaces that as console CTRL_C).
_LOCK_EMPTY_GRACE_SEC = 5.0


def _lock_is_stale(lock_path: str, *, now: Optional[float] = None) -> bool:
    """True when the lock may be unlinked and recreated by a waiter."""
    owner = _read_lock_owner_pid(lock_path)
    if owner is not None:
        return not _pid_is_alive(owner)
    try:
        age = (time.time() if now is None else now) - os.path.getmtime(lock_path)
    except OSError:
        return True
    return age > _LOCK_EMPTY_GRACE_SEC


def clear_stale_training_locks() -> list[str]:
    """Remove CUDA/opponent lock files whose owner PID is gone (or empty/corrupt).

    Empty locks older than ``_LOCK_EMPTY_GRACE_SEC`` are leftovers from older
    builds or crashed holders. A crash mid-``sync_opponent_model`` otherwise
    blocks the next run for the full lock timeout.
    """
    cleared: list[str] = []
    for path in (cuda_init_lock_path(), opponent_load_lock_path()):
        if not os.path.exists(path):
            continue
        if not _lock_is_stale(path):
            continue
        try:
            os.unlink(path)
            cleared.append(path)
        except OSError:
            pass
    return cleared


@contextlib.contextmanager
def _cross_process_file_lock(
    lock_path: str, *, timeout: float, label: str
) -> Iterator[None]:
    start = time.monotonic()
    fd: int | None = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            os.fsync(fd)
            break
        except FileExistsError:
            if _lock_is_stale(lock_path):
                try:
                    os.unlink(lock_path)
                    continue
                except OSError:
                    pass
            if time.monotonic() - start > timeout:
                owner = _read_lock_owner_pid(lock_path)
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for {label} lock "
                    f"({lock_path}, owner_pid={owner}). "
                    "Another training process may be stuck on the GPU."
                )
            time.sleep(0.25)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def warmup_cuda_if_needed(device: torch.device) -> None:
    """Eager CUDA context creation in the current process (helps lazy init on some Windows setups)."""
    if device.type != "cuda":
        raise RuntimeError(
            f"warmup_cuda_if_needed expected a CUDA device, got {device!r}. "
            "This codebase does not support CPU training."
        )
    with cuda_init_file_lock():
        torch.cuda.init()
        torch.zeros(1, device=device)
