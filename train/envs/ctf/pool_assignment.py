"""Trainee vs opponent assignment for CTF unit pools (static or per-episode race swap)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.agent.squad_constants import encode_race_vector  # noqa: F401 — re-export
from src.agent.squad_constants import include_race_enabled  # noqa: F401 — re-export


def get_level_races(unit_pools_config: List[Dict]) -> List[str]:
    """Unique race labels in first-seen order (stable, not sorted)."""
    races: List[str] = []
    seen: set[str] = set()
    for pool in unit_pools_config:
        race = str(pool.get("race", "")).lower().strip()
        if race and race not in seen:
            races.append(race)
            seen.add(race)
    return races


def alternate_trainee_race_enabled(training_conf: Dict) -> bool:
    env_conf = training_conf.get("env", {}) if isinstance(training_conf.get("env"), dict) else {}
    if "alternate_trainee_race" in training_conf:
        return bool(training_conf["alternate_trainee_race"])
    return bool(env_conf.get("alternate_trainee_race", False))


def player_races_for_episode(
    unit_pools_config: List[Dict],
    episode_counter: int,
    *,
    alternate: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(player_1_race, player_2_race)`` for the current episode assignment."""
    p1: Optional[str] = None
    p2: Optional[str] = None
    for pool in unit_pools_config:
        race = str(pool.get("race", "")).lower().strip()
        if not race:
            continue
        pid = pool_player_id(
            pool, unit_pools_config, episode_counter, alternate=alternate
        )
        if pid == 1 and p1 is None:
            p1 = race
        elif pid == 2 and p2 is None:
            p2 = race
        if p1 is not None and p2 is not None:
            break
    return p1, p2


def get_trainee_race(
    unit_pools_config: List[Dict],
    episode_counter: int,
    *,
    alternate: bool,
) -> Optional[str]:
    """Race controlled by the policy on this episode (None = legacy static ``agent`` flag)."""
    if not alternate:
        return None
    races = get_level_races(unit_pools_config)
    if len(races) < 2:
        return None
    # ``episode_counter`` is incremented at the start of ``reset()`` (1-based).
    return races[(max(episode_counter, 1) - 1) % len(races)]


def pool_player_id(
    pool: Dict,
    unit_pools_config: List[Dict],
    episode_counter: int,
    *,
    alternate: bool,
) -> int:
    """Return 1 for trainee squads, 2 for opponent squads."""
    trainee_race = get_trainee_race(unit_pools_config, episode_counter, alternate=alternate)
    pool_race = str(pool.get("race", "")).lower().strip()
    if trainee_race and pool_race:
        return 1 if pool_race == trainee_race else 2
    return 1 if pool.get("agent", False) else 2


def is_pool_trainee(
    pool: Dict,
    unit_pools_config: List[Dict],
    episode_counter: int,
    *,
    alternate: bool,
) -> bool:
    return pool_player_id(pool, unit_pools_config, episode_counter, alternate=alternate) == 1
