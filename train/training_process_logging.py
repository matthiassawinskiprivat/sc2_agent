"""Persist WARNING/ERROR (+ graft-related INFO) from training processes to disk.

Console-only loguru output is easy to miss. The health watcher reads
``logs/process_alerts.log`` so graft mismatches and other warnings survive.
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

PROCESS_ALERTS_FILENAME = "process_alerts.log"
_SINK_MARKER = "_sc2_process_alerts_sink_id"


def _is_alert_record(record: dict) -> bool:
    """Keep WARNING+ and a small set of graft / weight-load INFO lines."""
    if record["level"].no >= logger.level("WARNING").no:
        return True
    msg = str(record["message"]).lower()
    needles = (
        "graft",
        "shape mismatch",
        "missing key",
        "unexpected key",
        "load_state_dict",
        "incompatible",
        "rollout_policy",
    )
    return any(n in msg for n in needles)


def process_alerts_path(logs_dir: str = "logs") -> str:
    return os.path.join(logs_dir, PROCESS_ALERTS_FILENAME)


def configure_process_alert_logging(logs_dir: str = "logs") -> Optional[int]:
    """
    Attach a file sink for alerts. Safe to call in parent and child processes.

    Returns the sink id, or ``None`` if this interpreter already attached one.
    """
    if getattr(logger, _SINK_MARKER, None) is not None:
        return getattr(logger, _SINK_MARKER)

    os.makedirs(logs_dir, exist_ok=True)
    path = process_alerts_path(logs_dir)
    sink_id = logger.add(
        path,
        level="INFO",
        filter=_is_alert_record,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{process.name}:{process.id} | {name}:{function}:{line} | {message}"
        ),
        rotation="32 MB",
        retention=5,
        enqueue=True,
        encoding="utf-8",
    )
    setattr(logger, _SINK_MARKER, sink_id)
    logger.debug(f"Process alert logging attached -> {path}")
    return sink_id
