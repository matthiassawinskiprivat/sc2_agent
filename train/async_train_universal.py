import errno
import os
import subprocess
import sys
import time

# Must be set before the CUDA allocator first initialises (i.e. before any tensor.cuda() call).
# max_split_size_mb: prevents the allocator from splitting large blocks into small fragments
#   that can never be merged back, which causes the reserved pool to grow indefinitely.
# garbage_collection_threshold: proactively triggers GC when allocated fraction reaches
#   this proportion of the total reserved pool, reclaiming freed blocks earlier.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.8"

import torch
import numpy as np
import yaml
import json
import gc
from typing import Optional
from loguru import logger
import multiprocessing as mp
from gymnasium import spaces

# Add project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_PROJECT_ROOT)

COLLECTOR_ROLLOUT_HEARTBEAT_SEC = 30.0
LEARNER_BUFFER_WAIT_HEARTBEAT_SEC = 60.0


def heartbeat_due(last_ts: float, now: float, interval_sec: float) -> bool:
    """True when ``interval_sec`` has elapsed since ``last_ts``."""
    return (now - last_ts) >= float(interval_sec)


def collector_fill_stalled(last_progress_ts: float, now: float, timeout_sec: float) -> bool:
    """True when no rollout rows have been written for ``timeout_sec``."""
    return heartbeat_due(last_progress_ts, now, timeout_sec)


def cuda_empty_cache_requested(env_value: Optional[str], yaml_flag: bool) -> bool:
    """Opt-in only. Never implied by squad/CTF architecture (WDDM + live SC2)."""
    raw = (env_value or "").strip()
    return bool(yaml_flag) or raw in ("1", "true", "True")


DEFAULT_COLLECTOR_STEP_TIMEOUT_SEC = 120.0
COLLECTOR_VEC_PIPE_ERRORS = (
    EOFError,
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionError,
    OSError,
)


def resolve_collector_step_timeout(training_params: dict) -> float:
    """Seconds without env progress before the collector kills hung SC2 workers."""
    raw = training_params.get("collector_step_timeout")
    if raw is None:
        return DEFAULT_COLLECTOR_STEP_TIMEOUT_SEC
    return max(30.0, float(raw))


def unblock_stuck_collector_vec_env(vec_env) -> int:
    """
    Unblock a hung ``vec_env.step()`` / ``env_method`` by killing SC2 and Subproc workers.

    The collector thread is blocked in a pipe recv; only terminating the other end
    (or SC2, for in-process SyncVecEnv) makes that recv raise so recovery can run.
    """
    logger.error(
        "Collector: unblocking hung vec_env (kill SC2 + terminate Subproc workers)"
    )
    force_kill_sc2()
    terminated = 0
    raw_processes = getattr(vec_env, "processes", None)
    processes = list(raw_processes) if isinstance(raw_processes, (list, tuple)) else []
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
                terminated += 1
        except Exception as exc:
            logger.debug(f"Collector: worker terminate failed: {exc}")
        try:
            if process.is_alive():
                process.kill()
        except Exception as exc:
            logger.debug(f"Collector: worker kill failed: {exc}")
    return terminated

from train.train_universal import UniversalPipelineManager, BOT_REGISTRY
from src.sc2_bot_universal import SC2BotUniversal
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.maskable_ppo import MaskablePPO
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.universal_policy import UniversalActorCriticPolicy
from src.agent.models.squad_policy import SquadRecurrentPolicy
from src.agent.models.squad_network import SquadNetwork
from src.agent.models.unit_control_network import UnitControlNetwork
from src.utils.training_device import resolve_training_device, warmup_cuda_if_needed
from src.utils.rollout_buffer_factory import async_rollout_buffer_class
from src.utils.async_buffer_setup import (
    configure_collector_static_buffer,
    prepare_learner_placeholder_buffer,
)
from src.utils.async_buffer_paths import ensure_async_buffer_dir_env, resolve_async_buffer_dir
from src.agent.utils.async_shared_ram_buffer import AsyncSharedRamBuffer
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer
from train.collector_custom_metrics import publish_custom_metrics, record_info_scalars

import shutil


def _vram_log(label: str, logfile: str = "logs/vram_monitor.log") -> None:
    """Write a VRAM snapshot to a persistent file AND to the logger.

    Also appends an nvidia-smi per-process breakdown so Task Manager growth can be
    attributed to a concrete PID (PyTorch learner/collector/worker vs SC2.exe).
    """
    import torch as _t
    if not _t.cuda.is_available():
        return
    alloc_mb = _t.cuda.memory_allocated() / (1024 ** 2)
    reserved_mb = _t.cuda.memory_reserved() / (1024 ** 2)
    import time as _time
    stamp = _time.strftime("%H:%M:%S")
    line = (
        f"[{stamp}] pid={os.getpid()} "
        f"{label}: allocated={alloc_mb:.0f}MB reserved={reserved_mb:.0f}MB\n"
    )
    logger.info(f"[VRAM] {label}: allocated={alloc_mb:.0f}MB reserved={reserved_mb:.0f}MB")
    try:
        os.makedirs("logs", exist_ok=True)
        with open(logfile, "a") as _f:
            _f.write(line)
            # Attribute dedicated GPU memory by process — torch.cuda.* only sees this PID.
            if "after-empty_cache" in label or "post-train" in label:
                breakdown = _gpu_process_breakdown()
                if breakdown:
                    _f.write(breakdown)
    except Exception:
        pass


def _gpu_process_breakdown() -> str:
    """Return nvidia-smi compute-app lines, or '' if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Also count SC2 processes (DirectX VRAM often invisible to nvidia-smi compute-apps).
    try:
        from src.utils.sc2_cleanup import count_sc2_processes
        sc2_n = count_sc2_processes()
    except Exception:
        sc2_n = -1
    rows = [f"  gpu-proc: {ln}" for ln in lines]
    rows.append(f"  sc2_procs={sc2_n}")
    return "\n".join(rows) + "\n"

import psutil

# Async buffer: memmaps + payload.npz + hot-swap model zip every rollout — heavy disk I/O.
# Default to C:/temp (typically the OS SSD) for lowest latency.
# Override with env var SC2_ASYNC_BUFFER_DIR if you want a different location.
TEMP_DIR = resolve_async_buffer_dir()
# Collector loads from this path. Never read `latest_learner_model.zip` directly — the learner
# replaces that file; simultaneous open on Windows causes Errno 13 / sharing violations.
COLLECTOR_WEIGHTS_STAGING_ZIP = os.path.join(TEMP_DIR, "latest_learner_model_collector_staging.zip")
COLLECTOR_READY_MARKER = os.path.join(TEMP_DIR, "collector_ready.marker")
_COLLECTOR_STARTUP_TIMEOUT_SEC = 180.0
_NATIVE_ACCESS_VIOLATION_EXIT_CODES = frozenset({3221225477, -1073741819})


def _tensorboard_preflight() -> None:
    """Fail fast when TensorBoard cannot import (common: setuptools>=81 drops pkg_resources)."""
    try:
        import pkg_resources  # noqa: F401 — required by tensorboard<=2.20 on setuptools>=81
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "TensorBoard needs pkg_resources. Pin setuptools in the conda env: "
            "pip install 'setuptools<81'"
        ) from e
    import tensorboard  # noqa: F401


def apply_pipeline_rollout_steps(cfg: dict, pipeline_config: dict) -> dict:
    """Override squad ``n_steps`` from the pipeline root.

    CTF observations are large float maps. ``n_steps: 1024`` with 6 games
    allocates two rollout copies that do not fit in 32 GB, so Windows pages
    them to disk on every step. A pipeline ``training.n_steps`` replaces the
    per-level value for squad levels only.
    """
    if cfg.get("agent_architecture") != "squad":
        return cfg
    pipe_training = pipeline_config.get("training") if isinstance(pipeline_config, dict) else None
    if not isinstance(pipe_training, dict) or "n_steps" not in pipe_training:
        return cfg
    training = cfg.setdefault("training", {})
    if isinstance(training, dict):
        training["n_steps"] = int(pipe_training["n_steps"])
    return cfg


def resolve_async_log_dir(level_name: str, training_params: dict) -> str:
    """
    TensorBoard/SB3 log directory for one async level.

    Default: logs_universal/<level_name>/ (one TB run per level, easy to find in UI).
    Optional YAML override: training.tensorboard_subdir (e.g. \"async\" for legacy layout).
    """
    tb_subdir = training_params.get("tensorboard_subdir", "")
    if tb_subdir:
        return os.path.join("logs_universal", level_name, tb_subdir)
    return os.path.join("logs_universal", level_name)


def _truncate_logs_before_resume(
    level_name: str,
    config: dict,
    checkpoint_path: str,
) -> None:
    """
    Free TensorBoard's file locks, then drop metrics past the resume checkpoint.
    """
    from train.training_log_utils import truncate_logs_for_checkpoint_resume

    tb_port = int(os.environ.get("SC2_TENSORBOARD_PORT", "6006"))
    free_port(tb_port)
    log_dir = resolve_async_log_dir(level_name, config.get("training", {}))
    try:
        truncate_logs_for_checkpoint_resume(log_dir, checkpoint_path)
    except Exception as exc:
        logger.warning(
            f"Could not truncate resume logs in {log_dir} "
            f"(checkpoint={checkpoint_path!r}): {exc}"
        )


def detect_active_log_level(log_dir_base: str = "logs_universal") -> str | None:
    """Most recently touched curriculum folder under logs_universal (for TensorBoard pinning)."""
    log_dir = os.path.abspath(log_dir_base)
    if not os.path.isdir(log_dir):
        return None
    candidates = []
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        if os.path.isdir(path):
            candidates.append((os.path.getmtime(path), name))
    if not candidates:
        return None
    return max(candidates)[1]


def build_tensorboard_logdir_spec(
    log_dir_base: str = "logs_universal",
    active_level: str | None = None,
) -> str:
    log_dir = os.path.abspath(log_dir_base)
    if active_level:
        active_path = os.path.join(log_dir, active_level)
        return f"active:{active_path},all:{log_dir}"
    return log_dir


def build_viewer_tensorboard_logdir_spec(
    repo_root: str | None = None,
    *,
    universal_base: str = "logs_universal",
    build_base: str = "logs_build_zerg",
) -> str:
    """Combined ``--logdir_spec`` for unit-control + build agent logs.

    Used by ``view_tensorboard.bat`` so both curricula appear in one TB session.
    """
    root = os.path.abspath(repo_root or ".")
    parts: list[str] = []

    def _abs(base: str) -> str:
        return base if os.path.isabs(base) else os.path.join(root, base)

    def _append_tree(label: str, base: str) -> None:
        path = os.path.abspath(_abs(base))
        if not os.path.isdir(path):
            # Still register so TB watches the path once training creates it.
            parts.append(f"{label}:{path}")
            return
        active = detect_active_log_level(path)
        if active:
            parts.append(f"{label}_active:{os.path.join(path, active)}")
        parts.append(f"{label}:{path}")

    _append_tree("control", universal_base)
    _append_tree("build", build_base)
    return ",".join(parts)


def _tensorboard_port_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port(port: int) -> int:
    """Terminate any process listening on TCP *port*. Returns how many were stopped."""
    stopped = 0
    seen_pids = set()
    for conn in psutil.net_connections(kind="inet"):
        if not conn.laddr or conn.laddr.port != port:
            continue
        if conn.status != psutil.CONN_LISTEN:
            continue
        if conn.pid in seen_pids:
            continue
        seen_pids.add(conn.pid)
        try:
            proc = psutil.Process(conn.pid)
            logger.info(
                f"Freeing port {port}: stopping pid={proc.pid} ({proc.name()})"
            )
            proc.terminate()
            proc.wait(timeout=5)
            stopped += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    if stopped:
        time.sleep(1.0)
    return stopped


def _start_tensorboard(
    log_dir_base: str = "logs_universal",
    port: int = 6006,
    *,
    active_level: str | None = None,
) -> None:
    """Launch TensorBoard in the background if nothing is listening on *port*."""
    try:
        _tensorboard_preflight()
    except RuntimeError as e:
        logger.error(f"TensorBoard not started: {e}")
        return

    log_dir = os.path.abspath(log_dir_base)
    os.makedirs(log_dir, exist_ok=True)

    run_hint = (
        "In TensorBoard, enable the 'active' run in the left sidebar "
        "(or filter for your level, e.g. ctf_level_04)."
    )

    if _tensorboard_port_open(port):
        logger.info(
            f"TensorBoard already running at http://localhost:{port} — not starting a "
            f"second instance. {run_hint}"
        )
        return

    if os.environ.get("SC2_RESTART_TENSORBOARD", "").lower() in ("1", "true", "yes"):
        free_port(port)

    tb_log = os.path.join(log_dir, "tensorboard_startup.log")
    # When a curriculum level is active, pin it as a named run so CTF curves are easy to find.
    logdir_arg = build_tensorboard_logdir_spec(log_dir_base, active_level)
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir_spec",
        logdir_arg,
        "--port",
        str(port),
        "--bind_all",
        "--reload_interval",
        "15",
        # Avoid Windows read errors while the learner appends to events.out.tfevents.
        "--load_fast",
        "false",
    ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        with open(tb_log, "a", encoding="utf-8") as log_fp:
            log_fp.write(f"\n--- launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_fp.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        time.sleep(2.0)
        if proc.poll() is not None:
            logger.error(
                f"TensorBoard exited immediately (code {proc.returncode}). "
                f"See {tb_log}"
            )
            return
        if not _tensorboard_port_open(port):
            logger.error(
                f"TensorBoard process started but port {port} is not listening. See {tb_log}"
            )
            return
        logger.info(
            f"TensorBoard at http://localhost:{port} (logdir_spec={logdir_arg}). {run_hint}"
        )
    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")


def refresh_tensorboard(
    log_dir_base: str = "logs_universal",
    port: int = 6006,
    *,
    active_level: str | None = None,
) -> None:
    """
    Restart TensorBoard so new curriculum levels (e.g. level_attack_1) appear in the UI.

    TensorBoard started at pipeline begin often does not list runs created hours later
    under a new logs_universal/<level>/ folder until the server is restarted.
    """
    if os.environ.get("SC2_DISABLE_TENSORBOARD", "").lower() in ("1", "true", "yes"):
        return
    if os.environ.get("SC2_NO_TENSORBOARD_LEVEL_REFRESH", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    free_port(port)
    _start_tensorboard(log_dir_base, port, active_level=active_level)
    if active_level:
        logger.info(
            f"TensorBoard refreshed for level '{active_level}': "
            f"http://localhost:{port} — enable run '{active_level}' in the left sidebar "
            f"(logs: logs_universal/{active_level}/)"
        )


def force_kill_sc2():
    """Aggressively kills any SC2 and related processes at startup."""
    logger.info("Performing hard cleanup of StarCraft II processes...")
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if "starcraft" in name or "sc2" in name or "battle.net" in name:
                logger.debug(f"Killing process: {name} ({proc.pid})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def recover_collector_vec_env(vec_env, model, env_fns, num_envs, is_recurrent):
    """
    Rebuild SubprocVecEnv after a worker dies (SC2 crash, reset failure, pipe EOF).
    Keeps the collector alive so the learner does not abort the training session.
    """
    from train.train_pipeline import MaskableSubprocVecEnv, MaskableSyncVecEnv
    from train.env_utils import vec_env_obs_focus_tags, is_squad_pointer_action_space
    from train.squad_batch_inference import (
        SquadBatchRolloutHelper,
        ensure_collector_lstm_states,
    )
    from train.squad_pointer_rollout import SquadPointerRolloutHelper

    logger.warning(
        "Collector: SubprocVecEnv worker died — rebuilding all SC2 environments "
        "and continuing rollout collection."
    )
    unblock_stuck_collector_vec_env(vec_env)
    try:
        vec_env.close()
    except Exception as exc:
        logger.warning(f"Collector: vec_env.close() during recovery: {exc}")
    force_kill_sc2()
    time.sleep(5)
    new_vec = (
        MaskableSubprocVecEnv(env_fns, start_method="spawn")
        if num_envs > 1
        else MaskableSyncVecEnv(env_fns)
    )
    model.set_env(new_vec)
    if is_recurrent:
        ensure_collector_lstm_states(model, num_envs)
    reset_out = new_vec.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    squad_helper = (
        SquadPointerRolloutHelper(num_envs)
        if is_squad_pointer_action_space(new_vec.action_space)
        else SquadBatchRolloutHelper(num_envs)
    )
    last_dones = np.ones(num_envs, dtype=bool)
    if is_squad_pointer_action_space(new_vec.action_space):
        prev_focus_tags = np.full((num_envs,), -1, dtype=np.int64)
    else:
        prev_focus_tags = vec_env_obs_focus_tags(new_vec, num_envs)
    return new_vec, obs, squad_helper, last_dones, prev_focus_tags


def clear_temp_dir():
    """Wipes the async buffer directory entirely."""
    if os.path.exists(TEMP_DIR):
        logger.info(f"Cleaning up async buffer directory: {TEMP_DIR}")
        try:
            # We use a retry loop because Windows sometimes locks files
            for _ in range(3):
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
                if not os.path.exists(TEMP_DIR): break
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Startup cleanup failed: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)


def _clear_collector_ready_marker() -> None:
    try:
        os.remove(COLLECTOR_READY_MARKER)
    except OSError:
        pass


def _write_collector_ready_marker() -> None:
    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(COLLECTOR_READY_MARKER, "w", encoding="utf-8") as marker:
        marker.write(str(time.time()))


def _wait_for_collector_ready(collector: mp.Process, timeout_sec: float) -> tuple[bool, int | None]:
    """Return (ready, exit_code). ready=True when marker exists or collector still running past timeout."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not collector.is_alive():
            return False, collector.exitcode
        if os.path.isfile(COLLECTOR_READY_MARKER):
            return True, None
        time.sleep(0.25)
    if collector.is_alive():
        logger.warning(
            f"Collector still running after {timeout_sec:.0f}s without ready marker; proceeding anyway."
        )
        return True, None
    return False, collector.exitcode


def _cleanup_after_collector_spawn_failure(backoff_sec: float) -> None:
    """Release GPU handles before retrying collector spawn on Windows."""
    _clear_collector_ready_marker()
    gc.collect()
    # Avoid empty_cache by default — shared GPU with surviving workers.
    if os.environ.get("SC2_CUDA_EMPTY_CACHE", "").strip() in ("1", "true", "True"):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug(f"CUDA cache cleanup after collector spawn failure: {exc}")
    time.sleep(backoff_sec)


def _describe_collector_exit_code(exit_code: int | None) -> str:
    if exit_code in _NATIVE_ACCESS_VIOLATION_EXIT_CODES:
        return (
            "Windows access violation (0xC0000005) during collector startup — usually a CUDA "
            "driver/DLL race when multiple processes init the GPU. Retrying after cleanup."
        )
    if exit_code == 1:
        return (
            "Python exception in collector (see 'Collector subprocess CRASHED' traceback above)."
        )
    return f"Unexpected exit code {exit_code}."


def _is_transient_file_lock(err: OSError) -> bool:
    """True when another process (e.g. collector loading the zip) still holds the file."""
    if isinstance(err, PermissionError):
        return True
    win = getattr(err, "winerror", None)
    if win == 32:  # ERROR_SHARING_VIOLATION
        return True
    return err.errno in (errno.EACCES, errno.EPERM, errno.EBUSY)


def replace_file_retry(src: str, dst: str, *, retries: int = 50, delay_sec: float = 0.1) -> None:
    """
    Atomic replace (os.replace). Retries on Windows sharing violations (e.g. antivirus
    or a reader still closing a handle on dst).
    """
    last: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                if attempt == 0 or attempt % 10 == 9:
                    logger.debug(
                        f"replace_file_retry: dst busy ({dst}), attempt {attempt + 1}/{retries}"
                    )
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


def copy_file_retry(src: str, dst: str, *, retries: int = 50, delay_sec: float = 0.1) -> None:
    """
    Copy a file with retries on Windows sharing / antivirus locks (Errno 13, WinError 32).
    Overwrites dst when the copy succeeds.
    """
    last: OSError | None = None
    for attempt in range(retries):
        try:
            shutil.copyfile(src, dst)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                if attempt == 0 or attempt % 10 == 9:
                    logger.debug(
                        f"copy_file_retry: busy ({src!r} -> {dst!r}), attempt {attempt + 1}/{retries}"
                    )
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


def push_opponent_checkpoint_to_envs(vec_env, checkpoint_path: Optional[str] = None) -> None:
    """Load opponent weights inside each vec-env worker (Subproc-safe path string).

    Only CTF envs implement ``sync_opponent_model``; move/attack levels skip silently.
    """
    try:
        if not vec_env.has_attr("sync_opponent_model"):
            logger.debug("Collector: skipping opponent sync (env has no sync_opponent_model)")
            return
        if checkpoint_path:
            vec_env.env_method("sync_opponent_model", checkpoint_path)
        else:
            vec_env.env_method("sync_opponent_model")
    except Exception as e:
        logger.warning(f"Collector: opponent sync on env workers failed: {e}")


def stage_learner_zip_for_collector(src_zip: str, staging_zip: str) -> None:
    """
    Atomically publish a private copy for the collector: copy to *.part then replace staging.
    Avoids reading a zip the learner is still writing or replacing.
    """
    part = staging_zip + ".part"
    copy_file_retry(src_zip, part)
    replace_file_retry(part, staging_zip)


COLLECTOR_WEIGHT_RELOAD_POLL_SEC = 0.1


def _emit_collector_status(message: str) -> None:
    """Stdout + loguru — collector runs in a child process; print matches worker visibility."""
    print(f"[Collector] {message}", flush=True)
    logger.info(message)


def maybe_poll_collector_weights(
    weight_sync: "CollectorWeightSync",
    model,
    vec_env,
    poll_state: dict,
) -> None:
    """
    Optional mid-rollout mtime poll — **must not run during an active fill**.

    Reloading learner weights after ``rollout_policy.zip`` was snapshotted (or while
    ``old_log_prob`` rows are being written) desyncs PPO: learner evaluates W_zip
    while the buffer was collected with W_new → multi-nat ``rollout_log_prob_abs_err``.
    Prefer boundary-only ``try_reload`` before each fill; this helper no-ops when
    ``weight_sync.fill_frozen``.
    """
    if getattr(weight_sync, "fill_frozen", False):
        return
    now = time.perf_counter()
    if now - poll_state.get("last_poll", 0.0) < COLLECTOR_WEIGHT_RELOAD_POLL_SEC:
        return
    poll_state["last_poll"] = now
    weight_sync.try_reload(model, vec_env, sync_opponent=False, log_success=True)


def save_rollout_policy_atomic(model, rollout_policy_path: str) -> str:
    """
    Atomically publish collector policy weights for the learner.

    Returns ``rollout_policy_zip_checksum`` of the published zip (what the learner
    will actually load — not an in-memory GPU float reduction).
    """
    from src.utils.model_utils import rollout_policy_zip_checksum

    staging = os.path.join(
        os.path.dirname(rollout_policy_path) or ".",
        "rollout_policy_staging.zip",
    )
    model.save(staging)
    replace_file_retry(staging, rollout_policy_path)
    return rollout_policy_zip_checksum(rollout_policy_path)


class CollectorWeightSync:
    """
    Hot-reload learner weights into the collector process.

    Only safe at **rollout boundaries** (before a buffer fill begins). Mid-fill
    reloads break on-policy PPO because ``rollout_policy.zip`` / ``old_log_prob``
    would no longer match the collecting weights.
    """

    def __init__(self) -> None:
        self.last_model_load_time: Optional[float] = None
        self.fill_frozen: bool = False

    def try_reload(
        self,
        model,
        vec_env,
        *,
        sync_opponent: bool = False,
        log_success: bool = True,
    ) -> bool:
        if self.fill_frozen:
            return False
        updated_model_path = os.path.join(TEMP_DIR, "latest_learner_model.zip")
        if not os.path.exists(updated_model_path):
            return False
        try:
            mtime = os.path.getmtime(updated_model_path)
            if self.last_model_load_time is not None and mtime <= self.last_model_load_time:
                return False
            logger.debug(f"Collector staging learner checkpoint (mtime={mtime})...")
            from src.utils.model_utils import inspect_checkpoint_policy_weights

            src_ok, src_reason, src_bad = inspect_checkpoint_policy_weights(
                updated_model_path
            )
            if not src_ok:
                if src_reason == "non_finite":
                    logger.error(
                        "Collector skipped learner hot-swap: non-finite policy in "
                        f"{updated_model_path!r} (keys={src_bad})"
                    )
                elif src_reason.startswith("load_failed") or src_reason == "not_a_zip":
                    logger.warning(
                        "Collector skipped learner hot-swap: unreadable checkpoint "
                        f"({src_reason}) in {updated_model_path!r}"
                    )
                else:
                    logger.warning(
                        f"Collector skipped learner hot-swap: {src_reason} "
                        f"in {updated_model_path!r}"
                    )
                self.last_model_load_time = mtime
                return False

            stage_learner_zip_for_collector(
                updated_model_path, COLLECTOR_WEIGHTS_STAGING_ZIP
            )
            staged_ok, staged_reason, staged_bad = inspect_checkpoint_policy_weights(
                COLLECTOR_WEIGHTS_STAGING_ZIP
            )
            if not staged_ok:
                if staged_reason == "non_finite":
                    logger.error(
                        "Collector rejected staged weights: non-finite policy parameters "
                        f"(keys={staged_bad}) in {COLLECTOR_WEIGHTS_STAGING_ZIP!r}"
                    )
                else:
                    logger.warning(
                        "Collector rejected staged weights: "
                        f"{staged_reason} in {COLLECTOR_WEIGHTS_STAGING_ZIP!r}"
                    )
                self.last_model_load_time = mtime
                return False
            # Prefer policy-only load (strict=False + graft) over SB3 set_parameters:
            # learner/collector obs-space drift (e.g. missing enemy_encoder) must not
            # abort hot-swap with a Missing key RuntimeError.
            from src.utils.model_utils import load_policy_weights_from_zip

            ok, missing, unexpected = load_policy_weights_from_zip(
                model.policy,
                COLLECTOR_WEIGHTS_STAGING_ZIP,
                model.device,
                strict=False,
            )
            if not ok:
                raise RuntimeError(
                    f"no policy weights in {COLLECTOR_WEIGHTS_STAGING_ZIP!r}"
                )
            if missing:
                logger.warning(
                    "Collector hot-swap partial load missing="
                    f"{missing[:8]}{'…' if len(missing) > 8 else ''}"
                )
            if unexpected:
                logger.debug(
                    "Collector hot-swap ignored unexpected="
                    f"{unexpected[:8]}{'…' if len(unexpected) > 8 else ''}"
                )
            self.last_model_load_time = mtime
            if log_success:
                from src.utils.model_utils import read_checkpoint_timesteps

                timesteps = read_checkpoint_timesteps(updated_model_path)
                ts_suffix = (
                    f", timesteps={timesteps:,}" if timesteps is not None else ""
                )
                _emit_collector_status(
                    "loaded updated trainee checkpoint from learner "
                    f"({os.path.basename(updated_model_path)}{ts_suffix})"
                )
            if sync_opponent:
                push_opponent_checkpoint_to_envs(vec_env, COLLECTOR_WEIGHTS_STAGING_ZIP)
            return True
        except Exception as e:
            logger.error(f"Collector failed to load weights: {e}")
            return False


def set_parameters_with_retry(model, load_path: str, device: torch.device, *, retries: int = 12, delay_sec: float = 0.12) -> None:
    """SB3 set_parameters can still hit brief locks (AV); retry transient OS errors."""
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            model.set_parameters(load_path, device=device)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                if attempt == 0 or attempt % 4 == 3:
                    logger.debug(
                        f"set_parameters_with_retry: {load_path!r} busy, attempt {attempt + 1}/{retries}"
                    )
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


def remove_file_retry(path: str, *, retries: int = 50, delay_sec: float = 0.1) -> None:
    """Best-effort unlink with retries (Windows antivirus / other readers)."""
    if not os.path.exists(path):
        return
    last: OSError | None = None
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


class AsyncCollector(mp.Process):
    """
    PRODUCER: Plays StarCraft II and streams data to the disk buffer.
    It never stops unless the Learner is too slow.
    """
    def __init__(
        self,
        level_name,
        config,
        stop_event,
        model_path=None,
        resume_timesteps_from_save=False,
    ):
        super().__init__()
        self.level_name = level_name
        self.config = config
        self.stop_event = stop_event
        self.model_path = model_path
        # True only for same-level checkpoint resume; False for fresh start or curriculum (prior level weights).
        self.resume_timesteps_from_save = resume_timesteps_from_save

    def run(self):
        try:
            from train.training_process_logging import configure_process_alert_logging

            configure_process_alert_logging()
            self._run_inner()
        except Exception:
            import traceback
            logger.error(
                f"Collector subprocess CRASHED with unhandled exception:\n"
                + traceback.format_exc()
            )
            raise

    def _run_inner(self):
        logger.info(f"Collector started for level: {self.level_name}")
        # Initialize env and model using UniversalPipelineManager logic
        manager = UniversalPipelineManager(config_dir=self.config['config_dir'])
        
        # 1. Setup Env (Simplified from _train_level)
        training_params = self.config["training"]
        env_params = self.config["env"]
        AsyncBufferClass = async_rollout_buffer_class(training_params, env_params)
        train_device = resolve_training_device(training_params)
        warmup_cuda_if_needed(train_device)
        logger.info(f"Collector policy device: {train_device}")
        num_envs = training_params["num_envs"]
        
        # Use the bot specified in the environment or registry
        # DO NOT force SC2BotUniversal here, as it breaks multi-unit levels
        
        # Use existing factory to build env
        from train.train_pipeline import MaskableSubprocVecEnv, MaskableSyncVecEnv
        from train.env_utils import (
            EnvFactory,
            vec_env_obs_focus_tags,
            vec_env_requires_policy_action,
            resolve_level_agent_architecture,
            is_squad_pointer_action_space,
        )
        from train.squad_batch_inference import (
            SquadBatchRolloutHelper,
            ensure_collector_lstm_states,
            write_rollout_rows_to_buffer,
        )
        from train.opponent_peek_ipc import (
            OpponentPeekSharedMemory,
            hydrate_vec_spatial_obs,
        )
        from train.squad_pointer_rollout import (
            SquadPointerRolloutHelper,
            stack_info_action_masks,
            stack_info_requires_policy,
        )
        from train.train_pipeline import ENV_REGISTRY
        
        EnvClass = ENV_REGISTRY.get(env_params.get("env_class", "Level0Env"))
        # Default 32 matches universal MultiDiscrete spatial resolution; key optional in YAML.
        _grid = env_params.get("grid_size", 32)
        combined_conf = training_params.copy()
        combined_conf.update(env_params)
        combined_conf["is_training"] = True
        combined_conf["agent_architecture"] = resolve_level_agent_architecture(
            self.config, self.level_name
        )
        combined_conf["config_dir"] = self.config.get("config_dir", "config_universal")
        combined_conf["spatial_resolution"] = 32
        for key in (
            "active_layers",
            "active_tactical_layers",
            "active_unit_scalars",
            "action_heads",
            "categorical_heads",
        ):
            if key in self.config:
                combined_conf[key] = self.config[key]
        if "model" in self.config:
            combined_conf["model"] = self.config["model"]
        combined_conf["level_name"] = self.level_name
        env_fns = [
            EnvFactory(
                EnvClass, self.level_name, False, _grid, i, num_envs,
                None, None, combined_conf, env_params.get("max_steps", 300)
            )
            for i in range(num_envs)
        ]

        def _make_vec_env():
            return (
                MaskableSubprocVecEnv(env_fns, start_method="spawn")
                if num_envs > 1
                else MaskableSyncVecEnv(env_fns)
            )

        vec_env_ref = _make_vec_env()
        vec_env = vec_env_ref
        _emit_collector_status("vectorized environment ready")

        # 2. Setup Policy
        model_params = self.config.get("model", {})
        algo_name = model_params.get("algorithm", "MaskablePPO")
        rnn_type = model_params.get("rnn_type", "gru")
        hidden_size = model_params.get("policy_kwargs", {}).get("lstm_hidden_size", 512)
        is_recurrent_algo = algo_name == "MaskableRecurrentPPO"
        is_squad_arch = combined_conf["agent_architecture"] == "squad"
        
        AlgoClass = MaskableRecurrentPPO if is_recurrent_algo else MaskablePPO
        if is_squad_arch:
            if not is_recurrent_algo:
                raise ValueError("Squad pointer architecture requires MaskableRecurrentPPO")
            PolicyClass = SquadRecurrentPolicy
        else:
            PolicyClass = UniversalRecurrentPolicy if is_recurrent_algo else UniversalActorCriticPolicy
        
        policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
        policy_kwargs["spatial_resolution"] = 32
        if is_recurrent_algo:
            policy_kwargs["rnn_type"] = rnn_type
        fe_class = policy_kwargs.get("features_extractor_class")
        if is_squad_arch:
            if fe_class in (
                "src.agent.models.squad_network.SquadNetwork",
                SquadNetwork,
            ):
                policy_kwargs["features_extractor_class"] = SquadNetwork
            else:
                policy_kwargs["features_extractor_class"] = SquadNetwork
        elif fe_class in (
            "src.agent.models.unit_control_network.UnitControlNetwork",
            UnitControlNetwork,
        ):
            policy_kwargs["features_extractor_class"] = UnitControlNetwork

        algo_kwargs = {
            "policy": PolicyClass,
            "env": vec_env,
            "policy_kwargs": policy_kwargs,
            "n_steps": training_params["n_steps"],
            "batch_size": training_params["batch_size"],
            "gamma": model_params["gamma"],
            "device": train_device,
        }
        if is_recurrent_algo:
            algo_kwargs["rnn_type"] = rnn_type

        if self.model_path:
            logger.info(f"Loading weights from: {self.model_path}")
            _emit_collector_status(f"loading checkpoint {self.model_path!r}")
            load_kwargs = {
                "env": vec_env,
                "policy": PolicyClass,
                "policy_kwargs": policy_kwargs,
                "device": train_device,
                "custom_objects": {"policy_class": PolicyClass},
            }
            if is_recurrent_algo:
                load_kwargs["rnn_type"] = rnn_type
            from src.utils.model_utils import load_curriculum_checkpoint, load_squad_checkpoint

            if is_squad_arch:
                model = load_squad_checkpoint(
                    self.model_path,
                    algo_kwargs,
                    load_kwargs,
                    observation_space=vec_env.observation_space,
                    action_space=vec_env.action_space,
                    grid_size=_grid,
                    restore_optimizer=self.resume_timesteps_from_save,
                )
            else:
                model = load_curriculum_checkpoint(
                    AlgoClass,
                    self.model_path,
                    algo_kwargs,
                    load_kwargs,
                    vec_env.observation_space,
                    vec_env.action_space,
                    grid_size=_grid,
                    restore_optimizer=self.resume_timesteps_from_save,
                )
            from src.utils.model_utils import read_checkpoint_timesteps

            init_ts = read_checkpoint_timesteps(self.model_path)
            ts_suffix = f", timesteps={init_ts:,}" if init_ts is not None else ""
            _emit_collector_status(
                f"loaded initial checkpoint {self.model_path!r}{ts_suffix}"
            )
        else:
            _emit_collector_status("creating fresh policy on GPU")
            model = AlgoClass(**algo_kwargs)

        if isinstance(model, MaskableRecurrentPPO):
            ensure_collector_lstm_states(model, num_envs)

        if self.model_path:
            from src.utils.model_utils import apply_checkpoint_timesteps

            if self.resume_timesteps_from_save:
                applied = apply_checkpoint_timesteps(model, self.model_path)
                logger.info(
                    f"Collector: resumed timestep counter from checkpoint ({applied:,})."
                )
            else:
                model.num_timesteps = 0

        # One-time command-row priors at curriculum graft. Skip CONTINUE so a
        # mid-level resume does not double-boost. Applied before
        # initial_model.zip so the learner and opponent load the same logits.
        from src.utils.model_utils import (
            maybe_apply_loud_command_head_init,
            maybe_seed_pointer_command_rows,
        )

        # Pointer commands (attack_unit / heal_unit) inherit their spatial
        # counterpart's row first, then any YAML loud-command prior applies.
        maybe_seed_pointer_command_rows(
            model,
            self.config,
            resume=bool(self.resume_timesteps_from_save),
        )
        maybe_apply_loud_command_head_init(
            model,
            self.config,
            resume=bool(self.resume_timesteps_from_save),
        )

        fe = model.policy.features_extractor
        if hasattr(fe, "global_cnn"):
            c0 = fe.global_cnn[0]
            logger.info(f"Collector Model initialized (squad global CNN in_channels={c0.weight.shape[1]})")
        elif hasattr(fe, "cnn_trunk"):
            c0 = fe.cnn_trunk[0]
            logger.info(f"Collector Model initialized (global CNN in_channels={c0.weight.shape[1]})")
        elif hasattr(fe, "cnn") and len(fe.cnn) > 0:
            logger.info(f"Collector Model initialized with CNN input shape: {fe.cnn[0].weight.shape}")
        else:
            logger.info("Collector Model initialized (features extractor ready)")
            
        # Save initial weights for Learner to ensure perfect synchronization at start
        initial_path = os.path.join(TEMP_DIR, "initial_model.zip")
        model.save(initial_path)
        logger.info("Collector: Saved initial weights for Learner synchronization.")
        push_opponent_checkpoint_to_envs(vec_env, initial_path)
        _write_collector_ready_marker()
        _emit_collector_status("ready for learner")
        
        # 3. Static Double-Buffer Initialization
        # We manually create fixed directories buffer_0 and buffer_1
        buffer_idx = 0
        buffers = []
        for i in range(2):
            static_path = os.path.join(TEMP_DIR, f"buffer_{i}")
            # We must instantiate the buffer to create the structure
            buf = AsyncBufferClass(
                training_params["n_steps"],
                vec_env.observation_space,
                vec_env.action_space,
                n_envs=num_envs,
                temp_dir=TEMP_DIR,
                rnn_type=rnn_type,
                hidden_size=hidden_size,
                enable_rnn_sampling=is_recurrent_algo,
            )
            buf.instance_id = f"buffer_{i}"
            configure_collector_static_buffer(
                buf, vec_env.observation_space, static_path
            )
            buffers.append(buf)

        logger.info("Collector: Starting SC2 environments (vec_env.reset)...")
        try:
            _reset_out = vec_env.reset()
        except Exception as _reset_exc:
            logger.error(
                f"Collector: vec_env.reset() failed — SC2 likely crashed on startup. "
                f"Error: {_reset_exc}. "
                "Kill all SC2_x64.exe processes in Task Manager, then restart training."
            )
            vec_env.close()
            raise
        logger.info("Collector: SC2 environments ready. Entering main loop...")
        obs = _reset_out[0] if isinstance(_reset_out, tuple) else _reset_out
        reset_infos = (
            _reset_out[1]
            if isinstance(_reset_out, tuple) and len(_reset_out) > 1
            else None
        )
        squad_batch_helper = (
            SquadPointerRolloutHelper(num_envs)
            if is_squad_pointer_action_space(vec_env.action_space)
            else SquadBatchRolloutHelper(num_envs)
        )
        is_squad_pointer_rollout = isinstance(squad_batch_helper, SquadPointerRolloutHelper)
        if is_squad_pointer_rollout:
            self._prev_obs_focus_tags = np.full((num_envs,), -1, dtype=np.int64)
        else:
            self._prev_obs_focus_tags = vec_env_obs_focus_tags(vec_env, num_envs)
        last_action_masks = stack_info_action_masks(reset_infos, num_envs)
        last_requires_policy = stack_info_requires_policy(reset_infos, num_envs)
        peek_shm = None
        if is_squad_pointer_rollout:
            try:
                peek_shm = OpponentPeekSharedMemory(
                    num_envs, vec_env.observation_space
                )
                squad_batch_helper.bind_peek_shm(peek_shm, vec_env)
            except Exception as peek_exc:
                logger.warning(
                    f"Collector: opponent peek shared memory unavailable "
                    f"({peek_exc}); falling back to pickled peek."
                )
                peek_shm = None

        from src.utils.watchdog import Watchdog
        step_timeout = resolve_collector_step_timeout(training_params)

        stall_watchdog = Watchdog(timeout=step_timeout, recurring=True)

        def _on_collector_step_stall():
            logger.error(
                f"Collector: no rollout rows for {step_timeout:.0f}s "
                "(hung Subproc worker / SC2, or opponent-only steps that never fill). "
                "Killing SC2 and workers so vec_env.step can raise and recover."
            )
            stall_watchdog.dump_stack_trace()
            unblock_stuck_collector_vec_env(vec_env_ref)

        stall_watchdog.callback = _on_collector_step_stall

        def _timed_vec_step(actions):
            """Wall time of one parallel ``step`` (all workers), including the pipe."""
            profiling = (
                isinstance(squad_batch_helper, SquadPointerRolloutHelper)
                and squad_batch_helper.collector_profile_enabled()
            )
            t0 = time.perf_counter() if profiling else None
            stepped = vec_env.step(actions)
            if t0 is not None:
                squad_batch_helper.profile_add("step", time.perf_counter() - t0)
            return stepped

        def _hydrate_obs(obs, infos, dones=None):
            if peek_shm is None:
                return obs
            return hydrate_vec_spatial_obs(
                obs,
                infos,
                peek_shm,
                num_envs,
                reset_infos=getattr(vec_env_ref, "reset_infos", None),
                dones=dones,
            )

        def _recover_vec_env(vec_env, model):
            nonlocal vec_env_ref, last_action_masks, last_requires_policy, peek_shm
            new_vec, new_obs, new_helper, new_dones, new_focus = recover_collector_vec_env(
                vec_env, model, env_fns, num_envs, is_recurrent_algo
            )
            vec_env_ref = new_vec
            self.last_dones = new_dones
            self._prev_obs_focus_tags = new_focus
            last_action_masks = None
            last_requires_policy = None
            if peek_shm is not None and isinstance(new_helper, SquadPointerRolloutHelper):
                new_helper.bind_peek_shm(peek_shm, new_vec)
            return new_vec, new_obs, new_helper
        
        # Track metrics manually to avoid wrapper attribute errors
        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs)
        recent_ep_rewards = []
        recent_ep_lengths = []
        weight_sync = CollectorWeightSync()

        rollout_idx = 0
        try:
          while not self.stop_event.is_set():
            current_buffer = buffers[buffer_idx % 2]
            t_rollout_start = time.perf_counter()

            # Ensure learner is done with this buffer before overwriting
            consume_timeout = training_params.get("buffer_consumer_timeout", 300)
            if not current_buffer.wait_for_consumer(timeout=consume_timeout):
                logger.error("Learner hung? Collector pausing.")
                time.sleep(5)
                continue

            # Trainee/collector policy only — opponent sync is env-driven (every_k hard reset).
            # Runs here (rollout boundary), not immediately when the learner publishes weights.
            weight_sync.try_reload(model, vec_env, sync_opponent=False)

            # Freeze hot-reload for the entire fill. Mid-fill try_reload would collect
            # old_log_prob under W_new while the learner still loads a different zip.
            weight_sync.fill_frozen = True
            from src.utils.model_utils import policy_params_checksum

            fill_policy_checksum = policy_params_checksum(model.policy)
            rollout_policy_path = os.path.join(
                current_buffer.instance_dir, "rollout_policy.zip"
            )

            # FILL BUFFER
            current_buffer.reset()
            if is_squad_pointer_rollout and torch.cuda.is_available():
                from src.utils.cuda_memory import log_cuda_memory
                _vram_log(f"COLLECTOR rollout-start buffer={buffer_idx}")
                log_cuda_memory(train_device, f"Collector rollout-start (buffer={buffer_idx})")
            # Use class-based detection instead of attribute presence:
            # some model objects may carry compatibility attributes even when
            # running non-recurrent PPO.
            is_recurrent = isinstance(model, MaskableRecurrentPPO)
            if isinstance(vec_env.action_space, spaces.MultiDiscrete):
                dummy_actions = np.zeros(
                    (num_envs, len(vec_env.action_space.nvec)), dtype=np.int64
                )
            else:
                dummy_actions = np.zeros((num_envs,), dtype=np.int64)

            model.policy.set_training_mode(False)
            filled = 0
            n_rollout_steps = training_params["n_steps"]
            rollout_info_sums: dict = {}
            last_rollout_hb = t_rollout_start
            last_fill_progress_ts = t_rollout_start
            last_unblock_ts = 0.0
            stall_watchdog.start()
            try:
                while filled < n_rollout_steps:
                    now_fill = time.perf_counter()
                    if heartbeat_due(
                        last_rollout_hb, now_fill, COLLECTOR_ROLLOUT_HEARTBEAT_SEC
                    ):
                        logger.info(
                            f"Collector: filling buffer={buffer_idx} "
                            f"{filled}/{n_rollout_steps} "
                            f"({now_fill - t_rollout_start:.0f}s elapsed)"
                        )
                        last_rollout_hb = now_fill
                    if (
                        collector_fill_stalled(
                            last_fill_progress_ts, now_fill, step_timeout
                        )
                        and (now_fill - last_unblock_ts) >= 5.0
                    ):
                        logger.error(
                            f"Collector: buffer fill stalled at "
                            f"{filled}/{n_rollout_steps} for "
                            f"{now_fill - last_fill_progress_ts:.0f}s "
                            "(no rollout rows). Killing SC2 and workers."
                        )
                        stall_watchdog.dump_stack_trace()
                        unblock_stuck_collector_vec_env(vec_env_ref)
                        last_unblock_ts = now_fill
                    try:
                        if last_requires_policy is not None:
                            requires_policy = last_requires_policy
                        else:
                            requires_policy = vec_env_requires_policy_action(vec_env, num_envs)

                        if not requires_policy.any():
                            new_obs, rewards, dones, infos = _timed_vec_step(dummy_actions)
                            new_obs = _hydrate_obs(new_obs, infos, dones)
                            from_info = stack_info_action_masks(infos, num_envs)
                            if from_info is not None:
                                last_action_masks = from_info
                            from_req = stack_info_requires_policy(infos, num_envs)
                            if from_req is not None:
                                last_requires_policy = from_req
                            # Non-trainee squad steps (legacy cycling / chunk) are
                            # skipped for the training buffer (requires_policy=False).
                            # Their rewards are the acting owner's perspective and
                            # must not enter rollout/ep_rew_mean. Squad-pointer CTF
                            # never takes this branch: trainee+opponent share one step.
                            # Do NOT add them to episode_rewards — only count the length.
                            episode_lengths += 1
                            for i, done in enumerate(dones):
                                if done:
                                    squad_batch_helper.clear_env(i)
                                    recent_ep_rewards.append(episode_rewards[i])
                                    recent_ep_lengths.append(episode_lengths[i])
                                    episode_rewards[i] = 0
                                    episode_lengths[i] = 0
                                    if len(recent_ep_rewards) > 100:
                                        recent_ep_rewards.pop(0)
                                        recent_ep_lengths.pop(0)
                            obs = new_obs
                            self.last_dones = dones
                            continue

                        if not hasattr(self, "last_dones"):
                            self.last_dones = np.ones(num_envs, dtype=bool)

                        pre_step_obs = obs
                        with torch.no_grad():
                            resolution = squad_batch_helper.resolve_rollout(
                                vec_env=vec_env,
                                num_envs=num_envs,
                                obs=obs,
                                requires_policy=requires_policy,
                                model=model,
                                is_recurrent=is_recurrent,
                                last_dones=self.last_dones,
                                prev_focus_tags=self._prev_obs_focus_tags,
                                lstm_states=model._last_lstm_states if is_recurrent else None,
                                device=model.device,
                                action_masks=last_action_masks,
                            )

                        for env_idx, batch in resolution.squad_batch_actions.items():
                            vec_env.env_method(
                                "set_squad_batch_actions", batch, indices=[env_idx]
                            )

                        new_obs, rewards, dones, infos = _timed_vec_step(resolution.step_actions)
                        new_obs = _hydrate_obs(new_obs, infos, dones)
                        from_info = stack_info_action_masks(infos, num_envs)
                        if from_info is not None:
                            last_action_masks = from_info
                        from_req = stack_info_requires_policy(infos, num_envs)
                        if from_req is not None:
                            last_requires_policy = from_req
                    except COLLECTOR_VEC_PIPE_ERRORS as pipe_exc:
                        logger.error(
                            f"Collector: vec_env worker pipe error during rollout "
                            f"({type(pipe_exc).__name__}: {pipe_exc}). Recovering..."
                        )
                        vec_env, obs, squad_batch_helper = _recover_vec_env(vec_env, model)
                        stall_watchdog.reset()
                        last_fill_progress_ts = time.perf_counter()
                        last_unblock_ts = last_fill_progress_ts
                        continue

                    episode_rewards += rewards
                    episode_lengths += 1

                    if not hasattr(self, 'custom_metrics'):
                        self.custom_metrics = {}

                    for i, done in enumerate(dones):
                        info = infos[i]
                        record_info_scalars(
                            self.custom_metrics, rollout_info_sums, info
                        )

                        if done:
                            recent_ep_rewards.append(episode_rewards[i])
                            recent_ep_lengths.append(episode_lengths[i])
                            episode_rewards[i] = 0
                            episode_lengths[i] = 0
                            if len(recent_ep_rewards) > 100:
                                recent_ep_rewards.pop(0)
                                recent_ep_lengths.pop(0)

                    rows_written = write_rollout_rows_to_buffer(
                        buffer=current_buffer,
                        resolution=resolution,
                        pre_step_obs=pre_step_obs,
                        rewards=rewards,
                        num_envs=num_envs,
                        is_recurrent=is_recurrent,
                        fallback_lstm=model._last_lstm_states if is_recurrent else None,
                        last_dones=self.last_dones,
                        max_rows=n_rollout_steps - filled,
                    )

                    obs = new_obs
                    self.last_dones = dones
                    if is_recurrent:
                        model._last_lstm_states = resolution.updated_lstm
                        if not is_squad_pointer_rollout:
                            self._prev_obs_focus_tags = vec_env_obs_focus_tags(
                                vec_env, num_envs
                            )
                    filled += rows_written
                    if rows_written > 0:
                        stall_watchdog.reset()
                        last_fill_progress_ts = time.perf_counter()

                # CRITICAL: Compute GAE (Advantages & Returns) before sending to Learner
                # We need the value of the LAST observation to bootstrap
                with torch.no_grad():
                    try:
                        if last_action_masks is not None:
                            masks = last_action_masks
                        else:
                            masks = np.array(vec_env.env_method("action_masks"))
                    except COLLECTOR_VEC_PIPE_ERRORS as pipe_exc:
                        logger.error(
                            f"Collector: vec_env worker pipe error before GAE "
                            f"({type(pipe_exc).__name__}: {pipe_exc}). Recovering..."
                        )
                        vec_env, obs, squad_batch_helper = _recover_vec_env(vec_env, model)
                        stall_watchdog.reset()
                        if last_action_masks is not None:
                            masks = last_action_masks
                        else:
                            masks = np.array(vec_env.env_method("action_masks"))
                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    masks_tensor = torch.as_tensor(masks).to(model.device)

                    # We only need the value
                    if is_recurrent:
                        if is_squad_pointer_rollout:
                            lstm_starts_last = dones.astype(np.float32)
                        else:
                            focus_last = vec_env_obs_focus_tags(vec_env, num_envs)
                            lstm_starts_last = np.logical_or(
                                dones,
                                focus_last != self._prev_obs_focus_tags,
                            ).astype(np.float32)
                            self._prev_obs_focus_tags = focus_last.copy()
                        starts_tensor = torch.as_tensor(lstm_starts_last).to(model.device)
                        last_values = model.policy.predict_values(obs_tensor, model._last_lstm_states.vf, starts_tensor)
                    else:
                        last_values = model.policy.predict_values(obs_tensor)
                    fe = getattr(model.policy, "features_extractor", None)
                    if fe is not None and hasattr(fe, "clear_forward_cache"):
                        fe.clear_forward_cache()
            finally:
                stall_watchdog.stop()
                # Always unfreeze so a failed fill cannot permanently block hot-reload.
                weight_sync.fill_frozen = False
            
            current_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

            # FLUSH MEMMAPS TO DISK
            # Ensure the Learner sees the data we just wrote
            for key, mm in current_buffer.observations.items():
                if isinstance(mm, np.memmap): mm.flush()

            # Snapshot the *exact* weights that produced this buffer's old_log_prob.
            # Atomic replace avoids the learner reading a half-written zip. Saving
            # after the fill (still before any try_reload) pairs zip ↔ payload.
            end_checksum = policy_params_checksum(model.policy)
            if end_checksum != fill_policy_checksum:
                logger.error(
                    "Collector: policy checksum changed during fill "
                    f"({fill_policy_checksum} → {end_checksum}); "
                    "refusing to signal a desynced buffer"
                )
                continue
            saved_checksum = save_rollout_policy_atomic(model, rollout_policy_path)

            t_rollout_end = time.perf_counter()
            rollout_wall_sec = t_rollout_end - t_rollout_start
            filled_steps = int(current_buffer.pos)
            total_env_steps = filled_steps * training_params.get("num_envs", 1)
            instant_fps = total_env_steps / rollout_wall_sec if rollout_wall_sec > 0 else 0
            profile_note = ""
            profile_on = os.environ.get("SC2_COLLECTOR_PROFILE", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            means: dict = {}
            if profile_on and hasattr(vec_env, "has_attr"):
                try:
                    if vec_env.has_attr("pop_collector_profile_means"):
                        profiles = vec_env.env_method("pop_collector_profile_means")
                        merged: dict = {}
                        counts: dict = {}
                        for row in profiles or []:
                            if not isinstance(row, dict):
                                continue
                            for key, val in row.items():
                                merged[key] = merged.get(key, 0.0) + float(val)
                                counts[key] = counts.get(key, 0) + 1
                        means = {
                            key: merged[key] / max(counts[key], 1) for key in merged
                        }
                except Exception:
                    means = {}
            if profile_on and isinstance(squad_batch_helper, SquadPointerRolloutHelper):
                try:
                    means.update(squad_batch_helper.pop_collector_profile_means())
                except Exception:
                    pass
            if means:
                # Largest first so creep/dps spikes are visible in the console.
                timed = {
                    k: v
                    for k, v in means.items()
                    if k.endswith("_ms")
                }
                ranked = sorted(
                    timed.items(), key=lambda kv: kv[1], reverse=True
                )
                # Keep the line readable; full set still goes to TensorBoard.
                top = ranked[:18]

                def _fmt_profile_kv(k: str, v: float) -> str:
                    short = k.split("profile_", 1)[-1]
                    if short.endswith("_rate"):
                        return f"{short}={v:.2f}"
                    if short.endswith("_n"):
                        return f"{short}={v:.0f}"
                    return f"{short}={v:.1f}ms"

                profile_note = "  " + " ".join(
                    _fmt_profile_kv(k, v) for k, v in top
                )
                if len(ranked) > len(top):
                    profile_note += f"  (+{len(ranked) - len(top)} more)"
                cache_bits = []
                for label, key in (
                    ("tg", "collector/profile_cache_tg_rate"),
                    ("tt", "collector/profile_cache_tt_rate"),
                    ("wpn", "collector/profile_cache_wpn_rate"),
                    ("gmap", "collector/profile_cache_gmap_rate"),
                ):
                    if key in means:
                        cache_bits.append(f"{label}={means[key]:.2f}")
                if cache_bits:
                    profile_note += "  cache_hit[" + " ".join(cache_bits) + "]"
                # SB3 HumanOutputFormat truncates keys (~36 chars). Long
                # ``obs_L_allied_health`` tags collide → ValueError on dump.
                signal_profile = {
                    k: v
                    for k, v in means.items()
                    if len(k) <= 40
                    and "/profile_obs_L_" not in k
                }
            else:
                signal_profile = {}
            _emit_collector_status(
                f"Rollout #{rollout_idx} done: {total_env_steps} env-steps in "
                f"{rollout_wall_sec:.1f}s  ({instant_fps:.0f} fps this rollout)"
                f"{profile_note}"
            )
            rollout_idx += 1

            # SIGNAL LEARNER
            signal_meta = {
                "level": self.level_name,
                "steps": training_params["n_steps"],
                "filled_steps": filled_steps,
                "collector/rollout_wall_sec": rollout_wall_sec,
                "collector/instant_fps": instant_fps,
                "rollout_policy_checksum": saved_checksum,
            }
            signal_meta.update(signal_profile)
            
            if recent_ep_rewards:
                signal_meta["rollout/ep_rew_mean"] = np.mean(recent_ep_rewards)
                signal_meta["rollout/ep_len_mean"] = np.mean(recent_ep_lengths)
            
            # Env info scalars: rolling means, plus rare-event rollout sums
            # (heal_casts, move/attack/cloak/uncloak/aoe/load/unload commands). Mean of the last
            # 100 steps hides one-in-a-hundred casts.
            publish_custom_metrics(
                signal_meta,
                getattr(self, "custom_metrics", {}),
                rollout_info_sums,
            )

            current_buffer.signal_ready(signal_meta)

            if is_squad_pointer_rollout and torch.cuda.is_available():
                from src.utils.cuda_memory import log_cuda_memory
                _vram_log(f"COLLECTOR rollout-end buffer={buffer_idx}")
                log_cuda_memory(train_device, f"Collector rollout-end (buffer={buffer_idx})")

            # Opt-in only. Squad used to empty_cache after every rollout; on Windows
            # that returns the CUDA pool to WDDM while SC2 and sibling workers still
            # hold the GPU, then the next forward can hang with no log.
            _collector_clear = cuda_empty_cache_requested(
                os.environ.get("SC2_CUDA_EMPTY_CACHE"),
                bool(training_params.get("learner_empty_cache", False)),
            )
            if _collector_clear and torch.cuda.is_available():
                import gc as _gc
                _gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if is_squad_pointer_rollout:
                    _vram_log(f"COLLECTOR after-empty_cache buffer={buffer_idx}")

            buffer_idx += 1
        finally:
            if isinstance(squad_batch_helper, SquadPointerRolloutHelper):
                squad_batch_helper.close_peek_shm(unlink=True)
            vec_env.close()

class AsyncLearner:
    """
    CONSUMER: Watches for ready buffers and performs GPU updates.
    """
    def __init__(self, level_name, config, resume_timesteps_from_save=False, model_path=None, collector_proc=None):
        self.level_name = level_name
        self.config = config
        self.resume_timesteps_from_save = resume_timesteps_from_save
        self.model_path = model_path
        self.collector_proc = collector_proc  # mp.Process — used to detect collector death
        # Same-level interrupted resume only; curriculum transfer uses fresh Adam.
        self.restore_optimizer = resume_timesteps_from_save and bool(model_path)
        self.latest_model_path = os.path.join(TEMP_DIR, "latest_learner_model.zip")
        training_params = config.get("training", {})
        self.log_dir = resolve_async_log_dir(level_name, training_params)
        os.makedirs(self.log_dir, exist_ok=True)
        # Ensure model directory exists
        self.model_dir = os.path.join("models_universal", level_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.save_freq = config["training"].get("save_freq", 50000)

    def _check_collector_alive(self):
        """Raise RuntimeError if the collector subprocess has died unexpectedly."""
        if self.collector_proc is not None and not self.collector_proc.is_alive():
            exit_code = self.collector_proc.exitcode
            # exit code 1 = Python exception (check logs for full traceback)
            # 0xC0000005 / -1073741819 = SC2 access violation (kill SC2 processes)
            # 0xC0000409 / -1073740791 = Windows stack overrun (reboot)
            # 3221225477 = 0xC0000005 as unsigned
            native_crashes = {-1073741819, 3221225477, -1073740791, 3221226505}
            if exit_code in native_crashes:
                hint = "SC2 crashed (access violation). Kill all SC2_x64.exe in Task Manager, then restart."
            elif exit_code == 1:
                hint = (
                    "Python exception in collector (see 'Collector subprocess CRASHED' traceback above). "
                    "Common causes: WinError 1114 shm.dll = GPU DLL race (restart training); "
                    "ClientConnectionResetError = SC2 dropped WebSocket (restart training)."
                )
            else:
                hint = f"Unexpected exit code {exit_code}. Check console for collector traceback."
            raise RuntimeError(
                f"Collector subprocess died unexpectedly (exit code {exit_code}). {hint}"
            )

    def close(self):
        """Clean up handles to prevent WinError 32 leaks across curriculum levels"""
        if hasattr(self, 'logger'):
            try:
                self.logger.close()
            except Exception as e:
                logger.warning(f"Could not close logger: {e}")
        if hasattr(self, 'learner_buffers'):
            for buf in self.learner_buffers.values():
                if hasattr(buf, 'close_memmaps'):
                    buf.close_memmaps()

    def run(self):
        logger.info(f"Learner started for level: {self.level_name}")
        training_params = self.config["training"]
        env_params = self.config.get("env", {})
        AsyncBufferClass = async_rollout_buffer_class(training_params, env_params)
        train_device = resolve_training_device(training_params)
        warmup_cuda_if_needed(train_device)
        logger.info(
            f"Learner training device: {train_device} "
            f"(torch.cuda.is_available()={torch.cuda.is_available()})"
        )
        model_params = self.config.get("model", {})
        
        # 1. Setup Dummy Env for Model
        from train.env_utils import create_dummy_env_from_config, resolve_level_agent_architecture
        dummy_env = create_dummy_env_from_config(self.config, level_name=self.level_name)
        
        algo_name = model_params.get("algorithm", "MaskablePPO")
        rnn_type = model_params.get("rnn_type", "gru")
        hidden_size = model_params.get("policy_kwargs", {}).get("lstm_hidden_size", 512)
        is_recurrent_algo = algo_name == "MaskableRecurrentPPO"
        is_squad_arch = resolve_level_agent_architecture(self.config, self.level_name) == "squad"
        
        AlgoClass = MaskableRecurrentPPO if is_recurrent_algo else MaskablePPO
        if is_squad_arch:
            PolicyClass = SquadRecurrentPolicy
        else:
            PolicyClass = UniversalRecurrentPolicy if is_recurrent_algo else UniversalActorCriticPolicy
        
        policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
        policy_kwargs["spatial_resolution"] = 32
        if is_squad_arch:
            policy_kwargs["features_extractor_class"] = SquadNetwork
        elif "features_extractor_class" in policy_kwargs and policy_kwargs["features_extractor_class"] == "src.agent.models.unit_control_network.UnitControlNetwork":
            policy_kwargs["features_extractor_class"] = UnitControlNetwork

        algo_kwargs = {
            "policy": PolicyClass,
            "env": dummy_env,
            "n_steps": training_params["n_steps"],
            "batch_size": training_params["batch_size"],
            "n_epochs": model_params.get("n_epochs", 4),
            "gamma": model_params.get("gamma", 0.99),
            "learning_rate": training_params["learning_rate"],
            "ent_coef": model_params.get("ent_coef", 0.0),
            "policy_kwargs": policy_kwargs,
            "device": train_device,
            "tensorboard_log": self.log_dir,
            # Default None: run full n_epochs. YAML may set training.target_kl or model.target_kl.
            "target_kl": training_params.get("target_kl", model_params.get("target_kl")),
        }
        if AlgoClass == MaskableRecurrentPPO:
            algo_kwargs["rnn_type"] = rnn_type
            
        tk = algo_kwargs.get("target_kl")
        logger.info(
            f"Learner PPO target_kl={tk!r} (approx_kl early-stop at {1.5 * tk:.4f} per batch when set)"
            if tk is not None
            else "Learner PPO target_kl=None (no KL early-stop; full n_epochs each update)"
        )
        model = AlgoClass(**algo_kwargs)
        model.learner_profile = training_params.get("learner_profile", True)
        p0 = next(model.policy.parameters())
        logger.info(f"Learner model.device={model.device} policy params on {p0.device}")

        fe = model.policy.features_extractor
        if hasattr(fe, "global_cnn"):
            c0 = fe.global_cnn[0]
            logger.info(f"Learner Model initialized (squad global CNN in_channels={c0.weight.shape[1]})")
        elif hasattr(fe, "cnn_trunk"):
            c0 = fe.cnn_trunk[0]
            logger.info(f"Learner Model initialized (global CNN in_channels={c0.weight.shape[1]})")
        elif hasattr(fe, "cnn") and len(fe.cnn) > 0:
            logger.info(f"Learner Model initialized with CNN input shape: {fe.cnn[0].weight.shape}")
        else:
            logger.info("Learner Model initialized (features extractor ready)")
        
        # WAIT FOR COLLECTOR'S INITIAL WEIGHTS
        initial_path = os.path.join(TEMP_DIR, "initial_model.zip")
        logger.info("Learner: Waiting for initial weights from Collector...")
        total_steps_trained = 0
        for _ in range(120): # Wait up to 120 seconds (SC2 startup can be slow)
            self._check_collector_alive()
            if os.path.exists(initial_path):
                try:
                    from src.utils.model_utils import (
                        load_optimizer_state_from_zip,
                        load_policy_weights_from_zip,
                    )

                    try:
                        model.set_parameters(initial_path, device=train_device)
                    except Exception as init_exc:
                        ok, missing, _ = load_policy_weights_from_zip(
                            model.policy, initial_path, train_device, strict=False
                        )
                        if not ok or missing:
                            raise init_exc
                        logger.warning(
                            "Learner: set_parameters(initial) failed; loaded policy weights only "
                            f"(missing={len(missing)})"
                        )

                    if self.restore_optimizer:
                        opt_ok, opt_reason = load_optimizer_state_from_zip(
                            model, initial_path, train_device
                        )
                        if opt_ok:
                            logger.info("Learner: restored optimizer state from initial checkpoint.")
                        else:
                            logger.warning(
                                "Learner: optimizer restore skipped "
                                f"({opt_reason}); updates use a fresh Adam."
                            )

                    import zipfile
                    with zipfile.ZipFile(initial_path, 'r') as archive:
                        if self.resume_timesteps_from_save and 'data' in archive.namelist():
                            data_content = archive.read('data').decode('utf-8')
                            model_data = json.loads(data_content)
                            total_steps_trained = model_data.get("num_timesteps", 0)
                            model.num_timesteps = total_steps_trained
                            logger.info(f"Learner: Resuming from {total_steps_trained} timesteps.")
                        else:
                            total_steps_trained = 0
                            model.num_timesteps = 0
                            logger.info(
                                "Learner: Timestep counter at 0 for this level "
                                "(matches sync train_universal reset_num_timesteps on curriculum / fresh start)."
                            )

                    logger.info("Learner: Synchronized initial weights with Collector.")
                    break
                except Exception as e:
                    logger.warning(f"Learner failed to load initial weights: {e}")
            time.sleep(1)
        
        # Configure model logger
        from stable_baselines3.common.logger import configure
        new_logger = configure(self.log_dir, ["stdout", "tensorboard", "csv"])
        model.set_logger(new_logger)
        self.logger = new_logger
        # Create the TensorBoard run immediately so the level appears in the UI while the
        # collector is still gathering the first rollout (can take many minutes on CTF).
        # On same-level resume, dump at the restored step — not 0 — to avoid a fake jump.
        ready_step = total_steps_trained if self.resume_timesteps_from_save else 0
        new_logger.record("async/learner_ready", 1.0)
        new_logger.dump(step=ready_step)
        logger.info(f"Learner logging configured to: {self.log_dir}")

        # PRE-INITIALIZE LEARNER BUFFERS
        # We create persistent buffer objects that point to the static directories.
        # This prevents opening/closing file handles during the loop.
        self.learner_buffers = {}
        for i in range(2):
            instance_id = f"buffer_{i}"
            instance_dir = os.path.join(TEMP_DIR, instance_id)
            
            # Wait for collector to create the structure if it doesn't exist yet
            # (Collector creates them at startup)
            logger.info(f"Learner initializing static buffer: {instance_id}")
            
            # We construct the buffer manually to ensure it points to the right place
            # without creating NEW subdirectories
            buf = AsyncBufferClass(
                buffer_size=training_params["n_steps"],
                observation_space=dummy_env.observation_space,
                action_space=dummy_env.action_space,
                n_envs=training_params["num_envs"],
                temp_dir=TEMP_DIR,
                rnn_type=rnn_type,
                hidden_size=hidden_size,
                enable_rnn_sampling=is_recurrent_algo,
            )
            try:
                os.rmdir(buf.instance_dir)
            except OSError:
                pass
            buf.instance_id = instance_id
            buf.instance_dir = instance_dir
            prepare_learner_placeholder_buffer(buf)
            self.learner_buffers[instance_id] = buf

        logger.info("Learner waiting for first buffer...")
        start_time = time.time()
        last_buffer_wait_log = start_time
        buffer_wait_started = start_time
        
        while True:
            # Poll all possible buffer directories for a 'ready' signal
            found_buffer = False
            for d in os.listdir(TEMP_DIR):
                instance_dir = os.path.join(TEMP_DIR, d)
                if not os.path.isdir(instance_dir): continue
                
                signal_path = os.path.join(instance_dir, "buffer_ready.json")
                if os.path.exists(signal_path):
                    logger.info(f"Learner: Consuming buffer from {d}")
                    
                    try:
                        with open(signal_path, "r") as f:
                            meta = json.load(f)
                    except:
                        continue

                    # Update internal timestep for logging
                    filled = int(meta.get("filled_steps", meta["buffer_size"]))
                    num_steps = filled * meta["n_envs"]
                    total_steps_trained += num_steps
                    model.num_timesteps = total_steps_trained

                    # Calculate and log FPS
                    elapsed_time = time.time() - start_time
                    fps = int(total_steps_trained / elapsed_time) if elapsed_time > 0 else 0
                    model.logger.record("time/fps", fps)
                    model.logger.record("time/total_timesteps", total_steps_trained, exclude="tensorboard")
                    
                    # 1. SWITCH TO STATIC BUFFER
                    if d not in self.learner_buffers:
                        # Should not happen with static buffering 0/1
                        logger.warning(f"Unknown buffer ID {d}, skipping")
                        continue
                        
                    target_buffer = self.learner_buffers[d]
                    
                    # LOAD RAM PAYLOAD
                    payload_path = os.path.join(instance_dir, "payload.npz")
                    if os.path.exists(payload_path):
                        try:
                            # Always close NpzFile: leaving it open leaks Windows handles and keeps
                            # the archive locked, which worsens over long runs (slowdown + WinError 32).
                            with np.load(payload_path) as payload:
                                for key in ['actions', 'rewards', 'episode_starts', 'values', 'log_probs', 'advantages', 'returns', 'hidden_states_pi', 'hidden_states_vf']:
                                    try:
                                        setattr(target_buffer, key, payload[key].copy())
                                    except Exception as e:
                                        logger.error(f"Failed on key {key}: {getattr(target_buffer, key).flags}")
                                        raise e
                                        
                                if 'cell_states_pi' in payload and payload['cell_states_pi'].size > 0:
                                    target_buffer.cell_states_pi = payload['cell_states_pi'].copy()
                                    target_buffer.cell_states_vf = payload['cell_states_vf'].copy()
                                if 'action_masks' in payload and payload['action_masks'].size > 0:
                                    try:
                                        target_buffer.action_masks = payload['action_masks'].copy()
                                    except Exception as e:
                                        logger.error(f"Failed on action_masks: {target_buffer.action_masks.flags}")
                                        raise e
                                if 'policy_active' in payload and payload['policy_active'].size > 0:
                                    target_buffer.policy_active = payload['policy_active'].copy()

                                # Load RAM-based observations (e.g., 'command')
                                for key in payload.files:
                                    if key.startswith("obs_"):
                                        obs_key = key[4:]
                                        if obs_key in target_buffer.observations:
                                            try:
                                                if isinstance(target_buffer.observations[obs_key], np.memmap):
                                                    if not target_buffer.observations[obs_key].flags.writeable:
                                                        target_buffer.observations[obs_key].flags.writeable = True
                                                    target_buffer.observations[obs_key][:] = payload[key]
                                                else:
                                                    target_buffer.observations[obs_key] = payload[key].copy()
                                            except Exception as e:
                                                logger.error(f"Failed on obs {obs_key}: {target_buffer.observations[obs_key].flags}")
                                                raise e
                        except Exception as e:
                            logger.error(f"Failed to load buffer payload: {e}")
                            continue
                    else:
                        logger.warning(f"Payload not found for {d}, skipping")
                        continue

                    if meta.get("ipc_backend") == "shared_memory":
                        try:
                            AsyncSharedRamBuffer.attach_shared_observations(
                                target_buffer, meta
                            )
                        except Exception as exc:
                            logger.error(
                                f"Learner: shared obs attach failed ({exc}); "
                                "clearing signal so collector can refill"
                            )
                            try:
                                if os.path.exists(signal_path):
                                    os.remove(signal_path)
                            except OSError:
                                pass
                            continue
                    else:
                        try:
                            AsyncDiskBuffer.attach_disk_observations(
                                target_buffer, instance_dir, meta
                            )
                        except Exception as exc:
                            logger.error(
                                f"Learner: disk obs attach failed ({exc}); "
                                "clearing signal so collector can refill"
                            )
                            try:
                                if os.path.exists(signal_path):
                                    os.remove(signal_path)
                            except OSError:
                                pass
                            continue
                        from src.agent.utils.disk_buffer import is_disk_offloaded_obs_key

                        for key in dummy_env.observation_space.keys():
                            if is_disk_offloaded_obs_key(key):
                                continue
                            if key not in target_buffer.observations:
                                subspace = dummy_env.observation_space.spaces[key]
                                target_buffer.observations[key] = np.zeros(
                                    (
                                        meta["buffer_size"],
                                        meta["n_envs"],
                                        *subspace.shape,
                                    ),
                                    dtype=np.float32,
                                )
                        # Fingerprint already validated inside attach; mirror real result.
                        disk_obs = meta.get("disk_obs") or {}
                        model.logger.record(
                            "train/disk_obs_fingerprint_ok",
                            1.0 if disk_obs else 0.0,
                        )

                    model.rollout_buffer = target_buffer
                    model.rollout_buffer.full = True
                    model.rollout_buffer.pos = int(
                        meta.get("filled_steps", meta["buffer_size"])
                    )

                    rollout_policy_path = os.path.join(instance_dir, "rollout_policy.zip")
                    rollout_load_ok = 0.0
                    rollout_checksum_ok = 0.0
                    expected_checksum = meta.get("rollout_policy_checksum")
                    if os.path.exists(rollout_policy_path):
                        from src.utils.model_utils import (
                            load_policy_weights_from_zip,
                            rollout_policy_zip_checksum,
                        )

                        file_checksum = rollout_policy_zip_checksum(rollout_policy_path)
                        if expected_checksum and file_checksum != expected_checksum:
                            logger.error(
                                "Learner: rollout_policy.zip checksum mismatch "
                                f"(meta={expected_checksum} file={file_checksum}). "
                                "Skipping train — obs/logπ would be off-policy."
                            )
                            try:
                                if os.path.exists(signal_path):
                                    os.remove(signal_path)
                            except OSError:
                                pass
                            continue

                        ok, missing, unexpected = load_policy_weights_from_zip(
                            model.policy, rollout_policy_path, train_device, strict=False
                        )
                        if ok and not missing:
                            rollout_load_ok = 1.0
                            rollout_checksum_ok = (
                                1.0
                                if (not expected_checksum or file_checksum == expected_checksum)
                                else 0.0
                            )
                            logger.info(
                                "Learner: loaded rollout_policy.zip (collector snapshot for this buffer)"
                                + (f" checksum={file_checksum}" if file_checksum else "")
                            )
                        else:
                            if not ok:
                                logger.error(
                                    "Learner: rollout_policy.zip has no policy weights"
                                )
                            else:
                                logger.error(
                                    f"Learner: rollout_policy partial load missing={missing[:8]} "
                                    f"unexpected={unexpected[:8]}"
                                )
                    else:
                        logger.error(
                            f"Learner: rollout_policy.zip missing at {rollout_policy_path}"
                        )
                    
                    # Log ALL metrics passed from Collector
                    for key, val in meta.items():
                        if any(prefix in key for prefix in ["rollout/", "env/", "success"]):
                            model.logger.record(key, val)
                        elif key not in ["level", "steps", "buffer_size", "n_envs", "observation_space"]:
                            # Log anything else that looks like a metric
                            if isinstance(val, (int, float, np.number)):
                                model.logger.record(f"metrics/{key}", val)

                    model.logger.record("train/rollout_policy_load_ok", rollout_load_ok)
                    model.logger.record(
                        "train/rollout_policy_checksum_ok", rollout_checksum_ok
                    )
                    if rollout_load_ok < 0.5:
                        logger.error(
                            "Learner: refusing PPO update without a matching "
                            "rollout_policy.zip (would poison approx_kl / abs_err)"
                        )
                        try:
                            if os.path.exists(signal_path):
                                os.remove(signal_path)
                        except OSError:
                            pass
                        continue

                    # 2. PERFORM PPO UPDATE (+ timings: explains low GPU % = PCIe/Python vs kernel)
                    # Per-minibatch profiling: training.learner_profile in YAML (see parse_learner_profile_option).
                    t_learner_start = time.perf_counter()
                    logger.info("Learner: Starting GPU Update...")
                    try:
                        model.train()
                    finally:
                        # Always clear the ready signal after we bound this buffer, even if
                        # train() raises — otherwise the Collector waits until buffer_consumer_timeout.
                        try:
                            if os.path.exists(signal_path):
                                os.remove(signal_path)
                        except OSError as exc:
                            logger.warning(f"Learner: could not remove buffer signal {signal_path}: {exc}")

                    t_after_train = time.perf_counter()
                    if is_squad_arch and torch.cuda.is_available():
                        from src.utils.cuda_memory import log_cuda_memory
                        _vram_log("LEARNER post-train")
                        log_cuda_memory(train_device, "Learner post-train")
                    # empty_cache() forces a device sync and allocator churn; it can *lower* throughput
                    # and make GPU util look low while the CPU waits.
                    # Do not clear after squad/CTF train: collector workers + SC2 still hold the GPU.
                    # Returning the CUDA pool to WDDM lets those processes expand, then the next
                    # update reallocates into dedicated+shared memory. Opt-in via env/YAML.
                    _force_clear = cuda_empty_cache_requested(
                        os.environ.get("SC2_CUDA_EMPTY_CACHE"),
                        bool(training_params.get("learner_empty_cache", False)),
                    )
                    if _force_clear and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        if is_squad_arch:
                            _vram_log("LEARNER after-empty_cache")

                    # Force logger to write to disk
                    model.logger.dump(step=total_steps_trained)

                    # 3. Save updated weights for Collector (Hot Swap)
                    t_before_save = time.perf_counter()
                    from src.utils.model_utils import (
                        non_finite_state_dict_keys,
                        policy_state_dict_finite,
                    )

                    policy_sd = model.policy.state_dict()
                    weights_ok = policy_state_dict_finite(policy_sd)
                    if not weights_ok:
                        bad_keys = non_finite_state_dict_keys(policy_sd)
                        logger.error(
                            "Learner: non-finite policy weights after train(); "
                            f"skipping hot-swap publish (keys={bad_keys}). "
                            "Collector keeps previous weights."
                        )
                        if os.path.exists(self.latest_model_path):
                            from src.utils.model_utils import load_policy_weights_from_zip

                            restored, missing, _ = load_policy_weights_from_zip(
                                model.policy,
                                self.latest_model_path,
                                train_device,
                                strict=False,
                            )
                            if restored and not missing:
                                logger.warning(
                                    "Learner: restored policy from last good hot-swap "
                                    f"({self.latest_model_path})"
                                )
                            else:
                                logger.error(
                                    "Learner: could not restore last good weights; "
                                    "policy may stay non-finite until restart"
                                )
                    else:
                        tmp_save = self.latest_model_path + ".tmp"
                        model.save(tmp_save)
                        # os.replace overwrites dst; retries handle WinError 32 if collector
                        # still has latest_learner_model.zip open after set_parameters().
                        replace_file_retry(tmp_save, self.latest_model_path)
                    t_after_hot = time.perf_counter()

                    # 4. Save Checkpoint (Permanent Storage)
                    steps_before = total_steps_trained - num_steps
                    did_ckpt = (total_steps_trained // self.save_freq) > (steps_before // self.save_freq)
                    if did_ckpt and weights_ok:
                        ckpt_path = os.path.join(self.model_dir, f"universal_{total_steps_trained}_steps.zip")
                        model.save(ckpt_path)
                        logger.info(f"Saved checkpoint: {ckpt_path}")
                    elif did_ckpt:
                        logger.warning(
                            "Learner: skipped periodic checkpoint save (non-finite policy weights)"
                        )

                    t_end = time.perf_counter()
                    logger.info(
                        f"Learner timings: train={t_after_train - t_learner_start:.2f}s, "
                        f"hot_swap_save={t_after_hot - t_before_save:.2f}s, "
                        f"total={t_end - t_learner_start:.2f}s"
                    )

                    # Check for completion
                    if total_steps_trained >= training_params["training_steps"]:
                        final_path = os.path.join(self.model_dir, "final_model.zip")
                        model.save(final_path)
                        logger.info(f"Training completed! Saved final model: {final_path}")
                        try:
                            remove_file_retry(self.latest_model_path)
                        except OSError as e:
                            logger.warning(f"Could not remove hot-swap file {self.latest_model_path}: {e}")
                        return

                    if weights_ok:
                        from src.utils.model_utils import read_checkpoint_timesteps

                        published_ts = read_checkpoint_timesteps(self.latest_model_path)
                        ts_suffix = (
                            f", timesteps={published_ts:,}"
                            if published_ts is not None
                            else ""
                        )
                        logger.info(
                            "Learner: Update complete. Trainee checkpoint published "
                            f"({os.path.basename(self.latest_model_path)}{ts_suffix}). "
                            "Collector hot-reload runs after the in-flight rollout finishes."
                        )
                    else:
                        logger.warning(
                            "Learner: Update complete but trainee hot-swap was skipped "
                            "(non-finite weights); collector keeps the previous checkpoint."
                        )
                    found_buffer = True
                    last_buffer_wait_log = time.time()
                    buffer_wait_started = last_buffer_wait_log
                    break
            
            if not found_buffer:
                self._check_collector_alive()
                now_wait = time.time()
                if heartbeat_due(
                    last_buffer_wait_log, now_wait, LEARNER_BUFFER_WAIT_HEARTBEAT_SEC
                ):
                    waited = now_wait - buffer_wait_started
                    logger.info(
                        "Learner: still waiting for next collector buffer "
                        f"({waited:.0f}s). A couple of minutes is a normal CTF rollout; "
                        "longer usually means a Subproc worker is stuck in vec_env.step."
                    )
                    last_buffer_wait_log = now_wait
                time.sleep(1)


def run_single_async_training_session(
    level_name,
    config,
    model_to_load,
    resume_timesteps_from_save,
    collector_join_timeout=30.0,
):
    """
    One full async train (collector + learner) until training_steps for this level.
    Matches one call to sync UniversalPipelineManager._train_level for a single stage.
    """
    logger.info(f"--- [ASYNC] Target level: {level_name} ---")
    if model_to_load:
        logger.info(f"--- [ASYNC] Loading weights from: {model_to_load} ---")

    buffer_dir = ensure_async_buffer_dir_env()
    logger.info(f"Async buffer directory: {buffer_dir}")

    tb_port = int(os.environ.get("SC2_TENSORBOARD_PORT", "6006"))
    refresh_tensorboard("logs_universal", tb_port, active_level=level_name)

    force_kill_sc2()
    # Give Windows time to fully release resources/ports after SC2 is killed.
    # Skipping this pause causes the next SC2 launch to fail with access violations
    # when the previous process's network sockets and GPU resources are still in limbo.
    logger.info("Waiting 5 s for Windows to release SC2 resources before starting new processes...")
    time.sleep(5)

    # Verify no lingering SC2 processes remain (log a warning so the user knows to reboot).
    lingering = [p.info['pid'] for p in psutil.process_iter(['pid', 'name'])
                 if p.info.get('name', '').upper() in ('SC2_X64.EXE', 'SC2.EXE')]
    if lingering:
        logger.warning(
            f"SC2 processes still alive after kill attempt (PIDs: {lingering}). "
            "If training crashes immediately, reboot Windows to clear residual resources."
        )

    clear_temp_dir()

    stale_model = os.path.join(TEMP_DIR, "latest_learner_model.zip")
    if os.path.exists(stale_model):
        try:
            remove_file_retry(stale_model)
        except OSError:
            logger.warning(f"Could not remove stale model: {stale_model}")

    # Spawn the collector subprocess and wait until it publishes a ready marker
    # (model on GPU + initial_model.zip written). On Windows, CUDA DLL init can
    # race when the learner and collector start concurrently (0xC0000005).
    _COLLECTOR_SPAWN_RETRIES = 3
    stop_event = mp.Event()
    collector = None
    _clear_collector_ready_marker()
    for _spawn_attempt in range(_COLLECTOR_SPAWN_RETRIES):
        stop_event.clear()
        collector = AsyncCollector(
            level_name,
            config,
            stop_event,
            model_path=model_to_load,
            resume_timesteps_from_save=resume_timesteps_from_save,
        )
        collector.start()
        logger.info(
            f"Collector spawned (attempt {_spawn_attempt + 1}/{_COLLECTOR_SPAWN_RETRIES}). "
            f"Waiting up to {_COLLECTOR_STARTUP_TIMEOUT_SEC:.0f}s for ready marker..."
        )
        ready, exit_code = _wait_for_collector_ready(
            collector, _COLLECTOR_STARTUP_TIMEOUT_SEC
        )
        if ready and collector.is_alive():
            logger.info("Collector ready — starting learner.")
            break
        hint = _describe_collector_exit_code(exit_code)
        logger.error(
            f"Collector subprocess not ready (exit code {exit_code}). {hint} "
            + (
                f"Retrying after cleanup..."
                if _spawn_attempt < _COLLECTOR_SPAWN_RETRIES - 1
                else f"Collector startup failed after {_COLLECTOR_SPAWN_RETRIES} attempts."
            )
        )
        collector.join(timeout=5)
        if _spawn_attempt < _COLLECTOR_SPAWN_RETRIES - 1:
            _cleanup_after_collector_spawn_failure(5.0 * (_spawn_attempt + 1))
    else:
        raise RuntimeError(
            f"Collector subprocess failed to start after {_COLLECTOR_SPAWN_RETRIES} attempts. "
            "Check GPU drivers and CUDA installation. If exit code is 3221225477 "
            "(0xC0000005), reboot Windows to clear a stuck CUDA driver state."
        )

    learner = AsyncLearner(
        level_name,
        config,
        resume_timesteps_from_save=resume_timesteps_from_save,
        model_path=model_to_load,
        collector_proc=collector,
    )
    try:
        learner.run()
    except KeyboardInterrupt:
        logger.info("Shutdown signaled...")
        raise
    except RuntimeError as e:
        if "Collector subprocess died" in str(e):
            logger.error(str(e))
            raise
        raise
    finally:
        if 'learner' in locals() and hasattr(learner, 'close'):
            learner.close()
        stop_event.set()
        collector.terminate()
        collector.join(timeout=collector_join_timeout)
        if collector.is_alive():
            logger.warning("Collector did not exit in time; terminating.")
            try:
                collector.kill()
            except OSError:
                pass
        from src.utils.sc2_cleanup import kill_switch_sc2
        kill_switch_sc2(force=True)

        # Force Python GC and PyTorch CUDA cache clearing
        # Prevents VRAM fragmentation/exhaustion across curriculum levels
        if 'learner' in locals():
            del learner
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    from train.train_pipeline import cleanup_worker_logs
    from train.training_process_logging import configure_process_alert_logging

    configure_process_alert_logging()

    config_dir = "config_universal"
    manager = UniversalPipelineManager(config_dir=config_dir)
    cleanup_worker_logs()

    with open(os.path.join(config_dir, "pipeline.yaml"), "r") as f:
        pipeline_config = yaml.safe_load(f)

    from src.utils.config_manager import ConfigManager
    cm = ConfigManager(config_dir)

    # Set to a level name to train only that stage (single run), e.g. "level_move_multi_0"
    override_level = None

    def build_config_for_level(level_name: str):
        cfg = cm.get_level_config(level_name)
        if not cfg:
            yaml_path = os.path.join(config_dir, f"{level_name}.yaml")
            if not os.path.isfile(yaml_path):
                logger.error(
                    f"Missing level config (expected {yaml_path}, cwd={os.getcwd()})"
                )
            else:
                logger.error(
                    f"Level config is empty or invalid YAML: {yaml_path}"
                )
            return None
        cfg["active_layers"] = cm.get_cumulative_active_layers(level_name)
        cfg["active_tactical_layers"] = cm.get_cumulative_active_tactical_layers(level_name)
        cfg["active_unit_scalars"] = cm.get_cumulative_active_unit_scalars(level_name)
        cfg["action_heads"] = cm.get_cumulative_action_heads(level_name)
        cfg["categorical_heads"] = cm.get_cumulative_categorical_heads(level_name)
        cfg["agent_architecture"] = cm.get_agent_architecture(level_name)
        cfg["config_dir"] = config_dir
        apply_pipeline_rollout_steps(cfg, pipeline_config)
        return cfg

    if override_level:
        level_name = override_level
        status, model_path = manager.get_level_status(level_name)
        if status == "COMPLETED":
            logger.info(f"Override level '{level_name}' already has final_model. Nothing to do.")
            return
        resume_timesteps = status == "CONTINUE_TRAINING"
        if status == "CONTINUE_TRAINING":
            model_to_load = model_path
        elif status == "NEEDS_TRAINING":
            prev_final = None
            for lvl in pipeline_config["levels"]:
                if lvl == level_name:
                    model_to_load = prev_final
                    break
                st, pth = manager.get_level_status(lvl)
                if st == "COMPLETED":
                    prev_final = pth
        else:
            logger.error(f"Unexpected status for override level: {status}")
            sys.exit(1)

        config = build_config_for_level(level_name)
        if not config:
            logger.error(f"Could not load config for {level_name}")
            sys.exit(1)
        if status == "CONTINUE_TRAINING" and model_to_load:
            _truncate_logs_before_resume(level_name, config, model_to_load)
        try:
            run_single_async_training_session(
                level_name, config, model_to_load, resume_timesteps
            )
        except KeyboardInterrupt:
            pass
        return

    # Full pipeline: same order as UniversalPipelineManager.run() (sync train_universal.py)
    last_final_model_path = None
    for i in range(len(pipeline_config["levels"]) - 1, -1, -1):
        level = pipeline_config["levels"][i]
        status, path = manager.get_level_status(level)
        if status == "COMPLETED":
            last_final_model_path = path
            break

    for level_name in pipeline_config["levels"]:
        status, model_path = manager.get_level_status(level_name)
        if status == "COMPLETED":
            last_final_model_path = model_path
            continue

        config = build_config_for_level(level_name)
        if not config:
            logger.error(f"Could not load config for {level_name}")
            sys.exit(1)

        model_to_load = model_path if status == "CONTINUE_TRAINING" else last_final_model_path
        resume_timesteps = status == "CONTINUE_TRAINING"
        if status == "NEEDS_TRAINING":
            # Only wipe the async subdirectory, not the whole level log directory.
            # _cleanup_logs() deletes logs_universal/<level>/ and everything below it,
            # which includes the async/ subfolder the learner writes to.  When the
            # pipeline is restarted before a first checkpoint exists (NEEDS_TRAINING),
            # that call wipes the new run's event file on every restart — leaving only
            # the 88-byte TensorBoard header and an empty progress.csv.
            async_log_dir = resolve_async_log_dir(
                level_name, config.get("training", {})
            )
            if os.path.exists(async_log_dir):
                try:
                    shutil.rmtree(async_log_dir)
                    logger.info(f"Cleaned up async logs: {async_log_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean async log dir {async_log_dir}: {e}")
        elif status == "CONTINUE_TRAINING" and model_to_load:
            # Drop metrics logged after the last persistent *_steps.zip so TensorBoard
            # does not jump backwards when training resumes from that checkpoint.
            _truncate_logs_before_resume(level_name, config, model_to_load)

        logger.info(
            f"--- [ASYNC] Pipeline stage '{level_name}' "
            f"(status={status}, resume_ckpt_timesteps={resume_timesteps}) ---"
        )
        try:
            run_single_async_training_session(
                level_name, config, model_to_load, resume_timesteps
            )
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
            break

        last_final_model_path = os.path.join(
            manager.model_dir_base, level_name, "final_model.zip"
        )

    logger.info("Universal async pipeline finished (or stopped early).")


if __name__ == "__main__":
    main()
