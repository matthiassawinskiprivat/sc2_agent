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
    if len(h) > FIXED_UNIVERSAL_COMMAND_DIM:
        raise ValueError(
            f"action_heads length {len(h)} exceeds FIXED_UNIVERSAL_COMMAND_DIM={FIXED_UNIVERSAL_COMMAND_DIM}"
        )
    return h + [UNUSED_COMMAND_HEAD] * (FIXED_UNIVERSAL_COMMAND_DIM - len(h))
