"""Arena pairing helpers for CTF squad pointer stepping."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence


def trainee_pool_indices(
    unit_pools_config: Sequence[Dict],
    is_pool_trainee: Callable[[Dict], bool],
) -> List[int]:
    """Return pool indices owned by the trainee for the current episode."""
    return [i for i, pool in enumerate(unit_pools_config) if is_pool_trainee(pool)]


def paired_opponent_pool_index(
    trainee_pool_idx: int,
    unit_pools_config: Sequence[Dict],
    *,
    is_pool_trainee: Callable[[Dict], bool],
) -> Optional[int]:
    """
    Find the opponent pool in the same ``region`` as the trainee pool.

    Raises ``ValueError`` when pairing is ambiguous (not exactly one opponent).
    """
    if trainee_pool_idx < 0 or trainee_pool_idx >= len(unit_pools_config):
        return None

    trainee_pool = unit_pools_config[trainee_pool_idx]
    region = trainee_pool.get("region")
    if not region:
        raise ValueError(
            f"Trainee pool {trainee_pool_idx} missing 'region' for squad pairing"
        )

    candidates: List[int] = []
    for idx, pool in enumerate(unit_pools_config):
        if idx == trainee_pool_idx:
            continue
        if pool.get("region") != region:
            continue
        if is_pool_trainee(pool):
            continue
        candidates.append(idx)

    if len(candidates) != 1:
        raise ValueError(
            f"Ambiguous opponent pairing for trainee pool {trainee_pool_idx} "
            f"in region {region!r}: expected 1 opponent, got {candidates}"
        )
    return candidates[0]
