"""Unit selection resolver for squad pointer actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from src.agent.squad_constants import MAX_SQUAD_UNITS
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


@dataclass(frozen=True)
class SquadSelectionConfig:
    score_threshold: float = 0.5
    min_selected_units: int = 1
    max_selected_units: int | None = None


@dataclass(frozen=True)
class SquadSelectionResult:
    selected_indices: Tuple[int, ...]
    anchor_index: int
    scores: np.ndarray


def _clamp_selection(
    selected: List[int],
    scores: np.ndarray,
    alive_indices: List[int],
    *,
    min_selected: int,
    max_selected: int,
) -> Tuple[int, ...]:
    selected = sorted(set(selected))
    if len(selected) > max_selected:
        selected = sorted(selected, key=lambda i: scores[i], reverse=True)[:max_selected]
    if len(selected) < min_selected:
        ranked = sorted(alive_indices, key=lambda i: scores[i], reverse=True)
        selected = ranked[:min_selected]
    return tuple(int(i) for i in sorted(set(selected)))


def resolve_squad_selection(
    pointer_logits: Sequence[float] | np.ndarray,
    unit_mask: Sequence[float] | np.ndarray,
    *,
    config: SquadSelectionConfig | None = None,
) -> SquadSelectionResult:
    """Convert raw pointer logits into selected indices and anchor index."""
    cfg = config or SquadSelectionConfig()
    mask = np.asarray(unit_mask, dtype=np.float64).reshape(-1)
    logits = np.asarray(pointer_logits, dtype=np.float64).reshape(-1)
    if logits.shape[0] != mask.shape[0]:
        raise ValueError(
            f"pointer_logits length {logits.shape[0]} != unit_mask length {mask.shape[0]}"
        )

    scores = sigmoid(logits) * mask
    alive_indices = [i for i, m in enumerate(mask) if m > 0.5]
    if not alive_indices:
        empty = np.zeros_like(scores)
        return SquadSelectionResult(selected_indices=(), anchor_index=0, scores=empty)

    max_selected = cfg.max_selected_units
    if max_selected is None:
        max_selected = len(alive_indices)
    max_selected = max(0, min(int(max_selected), len(alive_indices)))
    min_selected = max(0, min(int(cfg.min_selected_units), max_selected))

    selected = [i for i in alive_indices if scores[i] >= cfg.score_threshold]
    selected = list(
        _clamp_selection(selected, scores, alive_indices, min_selected=min_selected, max_selected=max_selected)
    )
    anchor_index = max(alive_indices, key=lambda i: scores[i])
    return SquadSelectionResult(
        selected_indices=tuple(selected),
        anchor_index=int(anchor_index),
        scores=scores.astype(np.float32),
    )


def resolve_squad_selection_from_binary(
    pointer_binary: Sequence[int] | np.ndarray,
    unit_mask: Sequence[float] | np.ndarray,
    *,
    config: SquadSelectionConfig | None = None,
) -> SquadSelectionResult:
    """Resolve selection from sampled MultiDiscrete binary pointer slots."""
    cfg = config or SquadSelectionConfig()
    mask = np.asarray(unit_mask, dtype=np.float64).reshape(-1)
    binary = np.asarray(pointer_binary, dtype=np.int64).reshape(-1)
    if binary.shape[0] != mask.shape[0]:
        raise ValueError(
            f"pointer_binary length {binary.shape[0]} != unit_mask length {mask.shape[0]}"
        )
    scores = binary.astype(np.float32) * mask
    alive_indices = [i for i, m in enumerate(mask) if m > 0.5]
    if not alive_indices:
        return SquadSelectionResult(selected_indices=(), anchor_index=0, scores=scores)

    max_selected = cfg.max_selected_units
    if max_selected is None:
        max_selected = len(alive_indices)
    max_selected = max(0, min(int(max_selected), len(alive_indices)))
    min_selected = max(0, min(int(cfg.min_selected_units), max_selected))

    selected = [i for i in alive_indices if binary[i] > 0]
    selected = list(
        _clamp_selection(selected, scores, alive_indices, min_selected=min_selected, max_selected=max_selected)
    )
    anchor_index = max(alive_indices, key=lambda i: scores[i])
    return SquadSelectionResult(
        selected_indices=tuple(selected),
        anchor_index=int(anchor_index),
        scores=scores,
    )


def decode_squad_action(
    action: np.ndarray,
    unit_mask: Sequence[float] | np.ndarray,
    *,
    config: SquadSelectionConfig | None = None,
) -> Tuple[SquadSelectionResult, int, int]:
    """
    Split squad MultiDiscrete action into selection, command head, spatial index.

    Env action space layout (``MultiDiscrete([2]*N_max + [64, 1024])``):
    ``[pointer slot 0..N_max-1 | command | spatial]`` — length ``N_max + 2``.
    """
    action = np.asarray(action, dtype=np.int64).reshape(-1)
    expected = MAX_SQUAD_UNITS + 2
    if action.shape[0] != expected:
        raise ValueError(f"Expected action length {expected}, got {action.shape[0]}")

    pointer_binary = action[:MAX_SQUAD_UNITS]
    command_idx = int(action[MAX_SQUAD_UNITS])
    spatial_idx = int(action[MAX_SQUAD_UNITS + 1])
    selection = resolve_squad_selection_from_binary(
        pointer_binary, unit_mask, config=config
    )
    return selection, command_idx, spatial_idx


def selected_tags_from_indices(
    sorted_unit_tags: Sequence[int],
    selected_indices: Iterable[int],
) -> List[int]:
    """Map stable squad slot indices to SC2 unit tags."""
    tags = list(sorted_unit_tags)
    out: List[int] = []
    for idx in selected_indices:
        if 0 <= idx < len(tags):
            out.append(int(tags[idx]))
    return out
