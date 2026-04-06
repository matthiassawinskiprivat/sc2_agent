"""Shared rules for SC2 create_game player slots (Bot vs Computer)."""

from __future__ import annotations

from typing import Any, Dict

# Prefixes aligned with Level0Env / curriculum naming: movement maps must not run a
# Computer opponent (extra units, larvae, etc.) or map cleanup cannot empty the game.
MOVEMENT_LEVEL_PREFIXES = (
    "level_0",
    "level_1",
    "level_2",
    "level_3",
    "level_move_",
)


def use_computer_player(level_name: str, training_conf: Dict[str, Any]) -> bool:
    """
    Whether to add a Computer player alongside the training Bot.

    Default False for movement-style level names; override with env.use_computer.
    """
    is_movement_level = any(level_name.startswith(p) for p in MOVEMENT_LEVEL_PREFIXES)
    return training_conf.get("env", {}).get("use_computer", not is_movement_level)
