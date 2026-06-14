"""Shared rules for SC2 create_game player slots (Bot vs Computer)."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_EVERY_K_EPISODES = 100

# Prefixes aligned with Level0Env / curriculum naming: movement maps must not run a
# Computer opponent (extra units, larvae, etc.) or map cleanup cannot empty the game.
MOVEMENT_LEVEL_PREFIXES = (
    "level_0",
    "level_1",
    "level_2",
    "level_3",
    "level_move_",
)


def session_realtime(training_conf: Dict[str, Any]) -> bool:
    """
    Whether ``create_game`` / ``_prepare_start`` use realtime simulation.

    Training and debug-preview envs default to step-based sim (``False``) even when
    ``visualize=True``. Realtime breaks ``debug_create_unit`` / cleanup on multi-unit
    movement levels because ``client.step()`` does not advance the simulation the
    same way. Set ``env.realtime: true`` explicitly for human-facing realtime modes.
    """
    env = training_conf.get("env", {})
    if isinstance(env, dict) and "realtime" in env:
        return bool(env["realtime"])
    if "realtime" in training_conf:
        return bool(training_conf["realtime"])
    return False


def every_k_episodes(training_conf: Dict[str, Any]) -> int:
    """
    Episodes between scheduled hard resets (SC2 relaunch / map rotation).

    Read from flattened ``training_conf['every_k_episodes']`` or nested
    ``training_conf['env']['every_k_episodes']``. Defaults to
    ``DEFAULT_EVERY_K_EPISODES`` when unset.
    """
    env = training_conf.get("env", {})
    if "every_k_episodes" in training_conf:
        k = int(training_conf["every_k_episodes"])
    elif isinstance(env, dict) and "every_k_episodes" in env:
        k = int(env["every_k_episodes"])
    else:
        return DEFAULT_EVERY_K_EPISODES
    if k < 1:
        raise ValueError(f"every_k_episodes must be >= 1, got {k}")
    return k


def use_computer_player(level_name: str, training_conf: Dict[str, Any]) -> bool:
    """
    Whether to add a Computer player alongside the training Bot.

    Default False for movement-style level names; override with env.use_computer.
    """
    is_movement_level = any(level_name.startswith(p) for p in MOVEMENT_LEVEL_PREFIXES)
    return training_conf.get("env", {}).get("use_computer", not is_movement_level)
