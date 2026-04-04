"""
Resolve PyTorch device for SB3 training (sync universal + async learner/collector).

Precedence: env SC2_TRAINING_DEVICE > YAML training.device > auto.

``auto`` uses CUDA when available. Explicit ``cuda`` raises if CUDA is unavailable
so runs fail loudly instead of silently training on CPU.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import torch
from loguru import logger


def resolve_training_device(training_config: Optional[Mapping[str, Any]] = None) -> torch.device:
    cfg = training_config or {}
    raw = os.environ.get("SC2_TRAINING_DEVICE", "").strip().lower()
    if not raw:
        raw = str(cfg.get("device", "auto")).strip().lower()

    if raw in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Training device is set to CUDA (SC2_TRAINING_DEVICE or training.device) but "
                "torch.cuda.is_available() is False. Use a CUDA-enabled PyTorch build, check "
                "drivers, and CUDA_VISIBLE_DEVICES."
            )
        return torch.device("cuda:0")

    if raw == "cpu":
        return torch.device("cpu")

    if raw in ("auto", "", "default"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logger.warning(
            "CUDA is not available; training on CPU. Install a CUDA PyTorch wheel or fix drivers "
            "if you expected GPU. Set SC2_TRAINING_DEVICE=cuda to fail fast when CUDA is missing."
        )
        return torch.device("cpu")

    raise ValueError(
        f"Invalid training device {raw!r}. Use 'auto', 'cuda', or 'cpu' "
        "(SC2_TRAINING_DEVICE or training.device)."
    )


def warmup_cuda_if_needed(device: torch.device) -> None:
    """Eager CUDA context creation in the current process (helps lazy init on some Windows setups)."""
    if device.type != "cuda":
        return
    torch.cuda.init()
    torch.zeros(1, device=device)
