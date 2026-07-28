"""Truncate SB3 / TensorBoard training logs when resuming from a checkpoint."""

from __future__ import annotations

import csv
import os
import re
import time
from typing import Any, Optional

from loguru import logger


def resolve_keep_through_step(checkpoint_path: str) -> Optional[int]:
    """
    Timestep to keep in logs when resuming from ``checkpoint_path``.

    Prefers ``num_timesteps`` inside the SB3 zip; falls back to ``*_N_steps.zip``.
    """
    from src.utils.model_utils import read_checkpoint_timesteps

    ts = read_checkpoint_timesteps(checkpoint_path)
    if ts is not None:
        return int(ts)
    match = re.search(r"(\d+)_steps\.zip$", os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    return None


def _list_tfevent_files(log_dir: str) -> list[str]:
    return sorted(
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if "tfevents" in name
    )


def _remove_with_retry(path: str, *, retries: int = 30, delay_sec: float = 0.1) -> None:
    last_err: Optional[BaseException] = None
    for _ in range(retries):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_err = exc
            time.sleep(delay_sec)
    if last_err is not None:
        raise last_err


def _parse_csv_timestep(row: dict[str, str]) -> Optional[int]:
    raw = (row.get("time/total_timesteps") or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def truncate_progress_csv(log_dir: str, keep_through_step: int) -> dict[str, int]:
    """Keep ``progress.csv`` rows with empty or ``<= keep_through_step`` timesteps."""
    csv_path = os.path.join(log_dir, "progress.csv")
    stats = {"csv_rows_kept": 0, "csv_rows_dropped": 0}
    if not os.path.isfile(csv_path):
        return stats

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return stats
        kept_rows: list[dict[str, str]] = []
        for row in reader:
            step = _parse_csv_timestep(row)
            if step is None or step <= keep_through_step:
                kept_rows.append(row)
                stats["csv_rows_kept"] += 1
            else:
                stats["csv_rows_dropped"] += 1

    if stats["csv_rows_dropped"] == 0:
        return stats

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)
    os.replace(tmp_path, csv_path)
    return stats


def truncate_tensorboard_events(log_dir: str, keep_through_step: int) -> dict[str, int]:
    """
    Rewrite ``events.out.tfevents*`` keeping only scalar points with
    ``step <= keep_through_step``. Non-scalar event types are dropped.
    """
    stats = {
        "tb_scalars_kept": 0,
        "tb_scalars_dropped": 0,
        "events_rewritten": 0,
    }
    event_files = _list_tfevent_files(log_dir)
    if not event_files:
        return stats

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    size_guidance = {
        "scalars": 0,
        "tensors": 0,
        "histograms": 0,
        "images": 0,
        "audio": 0,
        "compressedHistograms": 0,
    }
    accumulator = EventAccumulator(log_dir, size_guidance=size_guidance)
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])

    kept_by_tag: dict[str, list[Any]] = {}
    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        kept = [event for event in events if event.step <= keep_through_step]
        dropped = len(events) - len(kept)
        stats["tb_scalars_kept"] += len(kept)
        stats["tb_scalars_dropped"] += dropped
        if kept:
            kept_by_tag[tag] = kept

    if stats["tb_scalars_dropped"] == 0:
        return stats

    for path in event_files:
        _remove_with_retry(path)

    writer = SummaryWriter(log_dir=log_dir)
    try:
        for tag, events in kept_by_tag.items():
            for event in events:
                writer.add_scalar(
                    tag,
                    event.value,
                    global_step=event.step,
                    walltime=event.wall_time,
                )
    finally:
        writer.close()

    stats["events_rewritten"] = 1
    return stats


def truncate_training_logs(log_dir: str, keep_through_step: int) -> dict[str, int]:
    """
    Remove CSV / TensorBoard metrics logged after ``keep_through_step``.

    Used on same-level resume so curves continue from the last persistent
    ``*_steps.zip`` instead of jumping back from post-checkpoint progress.
    """
    empty = {
        "csv_rows_kept": 0,
        "csv_rows_dropped": 0,
        "tb_scalars_kept": 0,
        "tb_scalars_dropped": 0,
        "events_rewritten": 0,
    }
    if keep_through_step < 0 or not os.path.isdir(log_dir):
        return empty

    csv_stats = truncate_progress_csv(log_dir, keep_through_step)
    tb_stats = truncate_tensorboard_events(log_dir, keep_through_step)
    return {**csv_stats, **tb_stats}


def truncate_logs_for_checkpoint_resume(
    log_dir: str,
    checkpoint_path: str,
) -> Optional[dict[str, int]]:
    """
    Truncate ``log_dir`` to the timestep of ``checkpoint_path``.

    Returns truncation stats, or ``None`` if the keep-through step cannot be
    resolved. No-ops (empty drops) still return stats.
    """
    keep_through = resolve_keep_through_step(checkpoint_path)
    if keep_through is None:
        logger.warning(
            f"Could not resolve resume timestep from {checkpoint_path!r}; "
            "leaving training logs unchanged."
        )
        return None

    stats = truncate_training_logs(log_dir, keep_through)
    dropped = stats["csv_rows_dropped"] + stats["tb_scalars_dropped"]
    if dropped:
        logger.info(
            f"Truncated training logs in {log_dir} to step <= {keep_through:,} "
            f"(csv_rows_dropped={stats['csv_rows_dropped']}, "
            f"tb_scalars_dropped={stats['tb_scalars_dropped']}, "
            f"checkpoint={os.path.basename(checkpoint_path)})"
        )
    else:
        logger.info(
            f"Resume log truncate: nothing past step {keep_through:,} in {log_dir}"
        )
    return stats
