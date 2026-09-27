"""Unit selection resolver for squad pointer actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from src.agent.squad_constants import (
    MAX_SQUAD_UNITS,
    SQUAD_ACTION_COMMAND_IDX,
    SQUAD_ACTION_DIM,
    SQUAD_ACTION_SPATIAL_IDX,
    SQUAD_ACTION_TARGET_IDX,
    target_slot_kind,
    target_slot_local_index,
    CMD_MODE_ALLY_TARGET,
    CMD_MODE_AOE,
    CMD_MODE_ATTACK_LOCAL,
    CMD_MODE_CLOAK,
    CMD_MODE_ENEMY_TARGET,
    CMD_MODE_LOAD_TARGET,
    CMD_MODE_SNIPE,
    CMD_MODE_SPATIAL,
    CMD_MODE_UNCLOAK,
    CMD_MODE_UNLOAD,
)

#: Observation actor-mask key for each command mode (same table the policy uses).
_ACTOR_MASK_KEY_BY_MODE = {
    CMD_MODE_ALLY_TARGET: "heal_actor_mask",
    CMD_MODE_LOAD_TARGET: "load_actor_mask",
    CMD_MODE_ENEMY_TARGET: "attack_actor_mask",
    CMD_MODE_ATTACK_LOCAL: "attack_actor_mask",
    CMD_MODE_AOE: "aoe_actor_mask",
    CMD_MODE_SNIPE: "snipe_actor_mask",
    CMD_MODE_UNLOAD: "unload_actor_mask",
    CMD_MODE_CLOAK: "cloak_actor_mask",
    CMD_MODE_UNCLOAK: "uncloak_actor_mask",
    CMD_MODE_SPATIAL: "move_actor_mask",
}


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


def eligible_actor_mask_from_obs(
    obs: dict | None,
    command_idx: int,
    *,
    n_slots: int = MAX_SQUAD_UNITS,
) -> np.ndarray | None:
    """Command-conditioned actor legality from the observation the policy saw.

    Returns ``None`` when the obs cannot supply a mask (missing ``command_modes``
    or unknown mode) so callers fall back to the alive-only clamp. A present
    all-zero array means “no capable actor” and must *not* be treated as missing.
    """
    if obs is None:
        return None
    modes = obs.get("command_modes")
    if modes is None:
        return None
    table = np.asarray(modes, dtype=np.float32).reshape(-1)
    idx = int(command_idx)
    if idx < 0 or idx >= table.shape[0]:
        return None
    mode = int(round(float(table[idx])))
    key = _ACTOR_MASK_KEY_BY_MODE.get(mode)
    if key is None:
        return None
    raw = obs.get(key)
    if raw is None:
        return None
    src = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(n_slots, dtype=np.float32)
    n = min(n_slots, src.shape[0])
    out[:n] = src[:n]
    return out


def _eligible_indices(
    alive_indices: List[int],
    eligible_mask: Sequence[float] | np.ndarray | None,
) -> List[int] | None:
    """Alive slots that may receive this command, or ``None`` to use all alive."""
    if eligible_mask is None:
        return None
    elig = np.asarray(eligible_mask, dtype=np.float64).reshape(-1)
    return [i for i in alive_indices if i < elig.shape[0] and elig[i] > 0.5]


def _clamp_selection(
    selected: List[int],
    scores: np.ndarray,
    pool_indices: List[int],
    *,
    min_selected: int,
    max_selected: int,
) -> Tuple[int, ...]:
    selected = sorted(set(selected))
    if len(selected) > max_selected:
        selected = sorted(selected, key=lambda i: scores[i], reverse=True)[:max_selected]
    if len(selected) < min_selected:
        ranked = sorted(pool_indices, key=lambda i: scores[i], reverse=True)
        selected = ranked[:min_selected]
    return tuple(int(i) for i in sorted(set(selected)))


def resolve_squad_selection(
    pointer_logits: Sequence[float] | np.ndarray,
    unit_mask: Sequence[float] | np.ndarray,
    *,
    config: SquadSelectionConfig | None = None,
    eligible_mask: Sequence[float] | np.ndarray | None = None,
) -> SquadSelectionResult:
    """Convert raw pointer logits into selected indices and anchor index.

    ``eligible_mask`` is the command-conditioned actor legality the policy used.
    Min/max clamp and the hard-anchor stay inside that pool. An empty eligible
    pool leaves the selection empty instead of backfilling an incapable unit.
    """
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

    pool = _eligible_indices(alive_indices, eligible_mask)
    if pool is not None and not pool:
        return SquadSelectionResult(
            selected_indices=(),
            anchor_index=0,
            scores=scores.astype(np.float32),
        )
    if pool is None:
        pool = alive_indices

    max_selected = cfg.max_selected_units
    if max_selected is None:
        max_selected = len(pool)
    max_selected = max(0, min(int(max_selected), len(pool)))
    min_selected = max(0, min(int(cfg.min_selected_units), max_selected))

    selected = [i for i in pool if scores[i] >= cfg.score_threshold]
    selected = list(
        _clamp_selection(selected, scores, pool, min_selected=min_selected, max_selected=max_selected)
    )
    anchor_index = max(pool, key=lambda i: scores[i])
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
    eligible_mask: Sequence[float] | np.ndarray | None = None,
) -> SquadSelectionResult:
    """Resolve selection from sampled MultiDiscrete binary pointer slots.

    When ``eligible_mask`` is given, min-select / anchor never pick a slot the
    policy was not allowed to turn on for this command.
    """
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

    pool = _eligible_indices(alive_indices, eligible_mask)
    if pool is not None and not pool:
        return SquadSelectionResult(selected_indices=(), anchor_index=0, scores=scores)
    if pool is None:
        pool = alive_indices

    max_selected = cfg.max_selected_units
    if max_selected is None:
        max_selected = len(pool)
    max_selected = max(0, min(int(max_selected), len(pool)))
    min_selected = max(0, min(int(cfg.min_selected_units), max_selected))

    selected = [i for i in pool if binary[i] > 0]
    selected = list(
        _clamp_selection(selected, scores, pool, min_selected=min_selected, max_selected=max_selected)
    )
    anchor_index = max(pool, key=lambda i: scores[i])
    return SquadSelectionResult(
        selected_indices=tuple(selected),
        anchor_index=int(anchor_index),
        scores=scores,
    )


@dataclass(frozen=True)
class SquadActionDecode:
    """Decoded squad action: actor selection, command, unit target, world cell."""

    selection: SquadSelectionResult
    command_idx: int
    target_slot: int
    spatial_idx: int


def decode_squad_action(
    action: np.ndarray,
    unit_mask: Sequence[float] | np.ndarray,
    *,
    config: SquadSelectionConfig | None = None,
    obs: dict | None = None,
) -> SquadActionDecode:
    """
    Split squad MultiDiscrete action into selection, command, target slot, cell.

    Env action space layout
    (``MultiDiscrete([2]*N_max + [64, SQUAD_TARGET_SLOTS, 1024])``):
    ``[actor bit 0..N_max-1 | command | target slot | spatial cell]`` —
    length ``N_max + 3``.

    When ``obs`` is the squad observation the policy saw, min-select clamp uses
    the same command-conditioned actor mask (``attack_actor_mask``, …).
    """
    action = np.asarray(action, dtype=np.int64).reshape(-1)
    if action.shape[0] != SQUAD_ACTION_DIM:
        raise ValueError(
            f"Expected action length {SQUAD_ACTION_DIM}, got {action.shape[0]}"
        )

    command_idx = int(action[SQUAD_ACTION_COMMAND_IDX])
    pointer_binary = action[:MAX_SQUAD_UNITS]
    eligible = eligible_actor_mask_from_obs(obs, command_idx)
    selection = resolve_squad_selection_from_binary(
        pointer_binary, unit_mask, config=config, eligible_mask=eligible
    )
    return SquadActionDecode(
        selection=selection,
        command_idx=command_idx,
        target_slot=int(action[SQUAD_ACTION_TARGET_IDX]),
        spatial_idx=int(action[SQUAD_ACTION_SPATIAL_IDX]),
    )


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


def target_tag_from_slot(
    target_slot: int,
    ally_tags: Sequence[int],
    enemy_tags: Sequence[int],
) -> int | None:
    """Resolve a sampled target slot to an SC2 unit tag (``None`` for null/out-of-range).

    ``ally_tags`` and ``enemy_tags`` must be the same sorted lists that filled the
    observation's ``unit_scalars`` / ``enemy_scalars`` rows, so slot identity is
    stable between the observation the policy saw and actuation.
    """
    kind = target_slot_kind(target_slot)
    local = target_slot_local_index(target_slot)
    if kind == CMD_MODE_ALLY_TARGET:
        pool = list(ally_tags)
    elif kind == CMD_MODE_ENEMY_TARGET:
        pool = list(enemy_tags)
    else:
        return None
    if 0 <= local < len(pool):
        return int(pool[local])
    return None
