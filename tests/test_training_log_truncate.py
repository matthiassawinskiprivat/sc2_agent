"""Tests for truncating TensorBoard / progress.csv logs on checkpoint resume."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from train.training_log_utils import (
    resolve_keep_through_step,
    truncate_logs_for_checkpoint_resume,
    truncate_training_logs,
)


def _write_progress_csv(log_dir: Path, timesteps: list[int | None]) -> None:
    path = log_dir / "progress.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["async/learner_ready", "time/total_timesteps", "train/loss"],
        )
        writer.writeheader()
        for step in timesteps:
            row = {
                "async/learner_ready": "1.0" if step is None else "",
                "time/total_timesteps": "" if step is None else str(step),
                "train/loss": "" if step is None else "0.5",
            }
            writer.writerow(row)


def _write_tb_scalars(log_dir: Path, points: list[tuple[str, int, float]]) -> None:
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for tag, step, value in points:
            writer.add_scalar(tag, value, global_step=step)
    finally:
        writer.close()


def _read_progress_timesteps(log_dir: Path) -> list[int | None]:
    path = log_dir / "progress.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[int | None] = []
    for row in rows:
        raw = (row.get("time/total_timesteps") or "").strip()
        out.append(None if not raw else int(float(raw)))
    return out


def _scalar_steps(log_dir: Path, tag: str) -> list[int]:
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return [event.step for event in ea.Scalars(tag)]


def test_resolve_keep_through_step_from_filename(tmp_path):
    ckpt = tmp_path / "universal_200704_steps.zip"
    ckpt.write_bytes(b"not-a-real-zip")
    # Corrupt / non-SB3 zip: filename fallback
    with patch(
        "src.utils.model_utils.read_checkpoint_timesteps",
        return_value=None,
    ):
        assert resolve_keep_through_step(str(ckpt)) == 200704


def test_resolve_keep_through_step_from_zip_data(tmp_path):
    ckpt = tmp_path / "universal_100_steps.zip"
    with zipfile.ZipFile(ckpt, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 200704}))
    assert resolve_keep_through_step(str(ckpt)) == 200704


def test_truncate_training_logs_drops_post_checkpoint_metrics(tmp_path):
    log_dir = tmp_path / "ctf_level_10"
    log_dir.mkdir()
    _write_progress_csv(log_dir, [None, 100352, 200704, 225000, 230000])
    _write_tb_scalars(
        log_dir,
        [
            ("train/loss", 100352, 1.0),
            ("train/loss", 200704, 0.8),
            ("train/loss", 225000, 0.7),
            ("train/loss", 230000, 0.6),
        ],
    )

    stats = truncate_training_logs(str(log_dir), keep_through_step=200704)

    assert stats["csv_rows_dropped"] == 2
    assert stats["tb_scalars_dropped"] == 2
    assert _read_progress_timesteps(log_dir) == [None, 100352, 200704]
    assert _scalar_steps(log_dir, "train/loss") == [100352, 200704]


def test_truncate_training_logs_noop_when_already_aligned(tmp_path):
    log_dir = tmp_path / "level"
    log_dir.mkdir()
    _write_progress_csv(log_dir, [100352, 200704])
    _write_tb_scalars(
        log_dir,
        [
            ("train/loss", 100352, 1.0),
            ("train/loss", 200704, 0.8),
        ],
    )
    before_events = list(log_dir.glob("*tfevents*"))
    assert before_events

    stats = truncate_training_logs(str(log_dir), keep_through_step=200704)

    assert stats["csv_rows_dropped"] == 0
    assert stats["tb_scalars_dropped"] == 0
    assert stats["events_rewritten"] == 0
    assert list(log_dir.glob("*tfevents*")) == before_events
    assert _read_progress_timesteps(log_dir) == [100352, 200704]


def test_truncate_logs_for_checkpoint_resume(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_progress_csv(log_dir, [150000, 200704, 210000])
    ckpt = tmp_path / "universal_200704_steps.zip"
    with zipfile.ZipFile(ckpt, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 200704}))

    stats = truncate_logs_for_checkpoint_resume(str(log_dir), str(ckpt))

    assert stats is not None
    assert stats["csv_rows_dropped"] == 1
    assert _read_progress_timesteps(log_dir) == [150000, 200704]
