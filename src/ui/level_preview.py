"""Training-level preview helpers for the debug UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.utils.config_manager import ConfigManager
from train.envs.sc2_session_config import session_realtime
from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    UNUSED_COMMAND_HEAD,
    build_action_head_manifest,
)

CONFIG_UNIVERSAL_DIR = "config_universal"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_pipeline_levels(config_dir: str = CONFIG_UNIVERSAL_DIR) -> List[str]:
    """Level names from ``pipeline.yaml`` (universal curriculum order)."""
    pipeline_path = PROJECT_ROOT / config_dir / "pipeline.yaml"
    if not pipeline_path.exists():
        return []
    with open(pipeline_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    levels = data.get("levels") or []
    return [str(name) for name in levels]


def build_preview_training_conf(level_name: str, config_dir: str = CONFIG_UNIVERSAL_DIR) -> Dict[str, Any]:
    """
    Build the same flattened ``training_conf`` shape ``Level0Env`` uses, plus preview flags.
    """
    cm = ConfigManager(config_dir)
    level_conf = cm.get_level_config(level_name)

    training_conf: Dict[str, Any] = {
        "config_dir": config_dir,
        "level_name": level_name,
        "preview_mode": True,
        "preview_computer_opponent": True,
        "agent_architecture": cm.get_agent_architecture(),
    }

    for section in ("training", "env"):
        if section in level_conf:
            for key, value in level_conf[section].items():
                training_conf[key] = value

    training_conf["active_layers"] = cm.get_cumulative_active_layers(level_name)
    training_conf["active_tactical_layers"] = cm.get_cumulative_active_tactical_layers(level_name)
    training_conf["action_heads"] = cm.get_cumulative_action_heads(level_name)
    training_conf["categorical_heads"] = cm.get_cumulative_categorical_heads(level_name)

    if training_conf.get("agent_architecture") == "universal":
        if not training_conf["active_layers"]:
            training_conf["active_layers"] = [
                "allied_unit_density",
                "enemy_unit_density",
                "terrain_pathable",
                "command_objective",
                "command_assignment",
            ]
        if not training_conf["active_tactical_layers"]:
            training_conf["active_tactical_layers"] = list(training_conf["active_layers"])
        if not training_conf["action_heads"]:
            training_conf["action_heads"] = ["move_global", "move_local"]
        training_conf["fixed_universal_command_dim"] = FIXED_UNIVERSAL_COMMAND_DIM
        training_conf["action_head_manifest"] = build_action_head_manifest(
            training_conf, list(training_conf["action_heads"])
        )

    env_section = level_conf.get("env", {})
    if "rewards" in env_section:
        training_conf["rewards"] = env_section["rewards"]
    if "unit_pools" in env_section:
        training_conf["unit_pools"] = env_section["unit_pools"]

    return training_conf


def discover_level_maps(level_name: str, training_conf: Dict[str, Any]) -> List[str]:
    """Maps available for a level (same resolution order as env ``_discover_maps``)."""
    config_maps_dir = training_conf.get("maps_dir") or training_conf.get("env", {}).get("maps_dir")
    if config_maps_dir:
        maps_dir = Path(config_maps_dir)
        if not maps_dir.is_absolute():
            maps_dir = PROJECT_ROOT / config_maps_dir
    else:
        maps_dir = PROJECT_ROOT / "train" / "maps" / level_name
    if not maps_dir.exists():
        return []
    return sorted(f.stem for f in maps_dir.glob("*.SC2Map"))


def resolve_env_class_name(level_name: str, config_dir: str = CONFIG_UNIVERSAL_DIR) -> str:
    level_conf = ConfigManager(config_dir).get_level_config(level_name)
    return level_conf.get("env", {}).get("env_class", "Level0Env")


def create_preview_env(
    level_name: str,
    map_name: str,
    training_conf: Optional[Dict[str, Any]] = None,
):
    """Instantiate the real training env with visualization enabled."""
    from train.train_pipeline import ENV_REGISTRY

    training_conf = dict(training_conf or build_preview_training_conf(level_name))
    env_class_name = resolve_env_class_name(level_name, training_conf.get("config_dir", CONFIG_UNIVERSAL_DIR))
    env_cls = ENV_REGISTRY.get(env_class_name)
    if env_cls is None:
        raise ValueError(f"Unknown env_class '{env_class_name}' for level '{level_name}'")

    kwargs: Dict[str, Any] = {
        "level_name": level_name,
        "visualize": True,
        "training_conf": training_conf,
        "worker_index": 0,
        "num_workers": 1,
        "max_steps": int(training_conf.get("max_steps", 500)),
    }

    # Step-based sim even when visualize=True (realtime breaks debug spawn / obs refresh).
    kwargs["realtime"] = session_realtime(training_conf)

    env = env_cls(**kwargs)
    env.map_name = map_name
    return env


def noop_action_for_env(env) -> Any:
    """Idle universal action that does not issue gameplay orders."""
    import numpy as np

    space = env.action_space
    if hasattr(space, "nvec"):
        manifest = (
            getattr(env, "training_conf", {}).get("action_head_manifest")
            or build_action_head_manifest(
                getattr(env, "training_conf", {}),
                list(getattr(env, "training_conf", {}).get("action_heads", [])),
            )
        )
        try:
            cmd = manifest.index(UNUSED_COMMAND_HEAD)
        except ValueError:
            cmd = 0
        return np.array([cmd, 0], dtype=np.int64)
    return 0


def push_obs_dict_to_agent(agent, obs: Any, training_conf: Dict[str, Any]) -> None:
    """Feed env observation tensors into the debug UI agent."""
    if isinstance(obs, dict):
        agent.ui_observation = obs.get("map")
        agent.tactical_observation = obs.get("tactical")
    else:
        agent.ui_observation = obs
        agent.tactical_observation = None
    agent.active_layers = training_conf.get("active_layers")
    agent.active_tactical_layers = training_conf.get("active_tactical_layers")


def sync_preview_debug_click_overlay(env, ui_agent) -> None:
    """
    Training-level preview: mirror SC2 unit move orders onto the debug UI agent.

    The env bot builds observations internally; the UI uses a separate ``main_agent``.
    Without this sync, ``last_debug_click`` stays empty and the green crosshair never draws.
    """
    from src.sc2_bot import resolve_tactical_active_unit_for_debug, sync_agent_debug_click_overlay

    bot = getattr(env, "_bot", None)
    if bot is None:
        return
    game_info = getattr(bot, "game_info", None)
    if game_info is not None:
        ui_agent.last_game_info = game_info
    active_unit, _ = resolve_tactical_active_unit_for_debug(bot)
    sync_agent_debug_click_overlay(ui_agent, active_unit, cache=bot)


def reconfigure_hierarchical_system(system, level_name: str, config_dir: str = CONFIG_UNIVERSAL_DIR) -> Dict[str, Any]:
    """Point the shared observation transformer at a level's cumulative layer config."""
    training_conf = build_preview_training_conf(level_name, config_dir=config_dir)
    system.training_conf = training_conf
    system.shared_transformer = type(system.shared_transformer)(
        map_size=(128, 128),
        training_conf=training_conf,
    )
    return training_conf
