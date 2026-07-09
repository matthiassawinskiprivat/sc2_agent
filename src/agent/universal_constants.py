"""Constants shared by universal env, policies, and bots."""

from typing import Dict, List, Any

# Fixed first axis of universal MultiDiscrete([N, 32*32]); unused slots are masked.
FIXED_UNIVERSAL_COMMAND_DIM = 64

# Sentinel for padded command slots (never executed, always masked illegal).
UNUSED_COMMAND_HEAD = "_unused"


def build_action_head_manifest(
    training_conf: Dict[str, Any], action_heads: List[str]
) -> List[str]:
    """
    Return a length-FIXED_UNIVERSAL_COMMAND_DIM manifest, reusing training_conf if valid.
    """
    existing = training_conf.get("action_head_manifest")
    if existing and len(existing) == FIXED_UNIVERSAL_COMMAND_DIM:
        return list(existing)
    h = list(action_heads)
    cats = list(training_conf.get("categorical_heads") or [])
    merged = h + cats
    if len(merged) > FIXED_UNIVERSAL_COMMAND_DIM:
        raise ValueError(
            f"action_heads + categorical_heads length {len(merged)} exceeds "
            f"FIXED_UNIVERSAL_COMMAND_DIM={FIXED_UNIVERSAL_COMMAND_DIM}"
        )
    return merged + [UNUSED_COMMAND_HEAD] * (FIXED_UNIVERSAL_COMMAND_DIM - len(merged))


# Tactical center cell (gx=16, gy=15) — matches SpatialDecoder bias / ObservationTransformer.
UNIVERSAL_DEFAULT_SPATIAL_IDX = 15 * 32 + 16


def universal_fallback_action() -> "np.ndarray":
    """Safe move_global + center cell when opponent/policy inference is unavailable."""
    import numpy as np

    return np.array([0, UNIVERSAL_DEFAULT_SPATIAL_IDX], dtype=np.int64)
