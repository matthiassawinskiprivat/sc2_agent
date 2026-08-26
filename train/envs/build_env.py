"""Gym environment for build-agent training (SC2 lifecycle from Level0Env)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from gymnasium import spaces

from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    BUILD_PLACEMENT_CELLS,
    COMMAND_VECTOR_DIM,
    MACRO_VECTOR_DIM,
    MAX_BUILD_WORKERS,
    WORKER_FEAT_DIM,
)
from src.utils.config_manager import ConfigManager
from src.zerg_build_bot import ZergBuildBot
from train.envs.level_0_env import Level0Env


class BuildEnv(Level0Env):
    """Build-agent env: separate obs/action contract from unit-control universal stack."""

    def __init__(self, *args, training_conf: Dict | None = None, **kwargs):
        training_conf = dict(training_conf or {})
        level_name = kwargs.get("level_name", "level_build_zerg_0")
        self._build_pre_init(training_conf, level_name)
        super().__init__(*args, training_conf=training_conf, **kwargs)
        self.agent_architecture = "build"
        self.training_conf["agent_architecture"] = "build"
        self._bot_cls = ZergBuildBot
        self._define_build_spaces()

    def _build_pre_init(self, training_conf: Dict, level_name: str) -> None:
        config_dir = training_conf.get("config_dir", "config_build_zerg")
        cm = ConfigManager(config_dir)
        level_conf = cm.get_level_config(level_name)
        for section in ("training", "env"):
            if section in level_conf:
                for k, v in level_conf[section].items():
                    training_conf[k] = v
        training_conf["build_map_layers"] = cm.get_cumulative_build_map_layers(level_name)
        training_conf["build_commands"] = cm.get_cumulative_build_commands(level_name)
        training_conf["config_dir"] = config_dir
        training_conf["level_name"] = level_name
        training_conf["race"] = training_conf.get("race", "zerg")
        env_section = training_conf.setdefault("env", {})
        if isinstance(env_section, dict):
            env_section.setdefault("maps_dir", "train/maps/ladder/build_agent")
        if "rewards" not in training_conf and "env" in level_conf:
            training_conf["rewards"] = level_conf["env"].get("rewards", {})

    def _define_build_spaces(self) -> None:
        n_map_ch = len(self.training_conf.get("build_map_layers", [])) or 3
        self.observation_space = spaces.Dict(
            {
                "local_map": spaces.Box(
                    0.0,
                    1.0,
                    shape=(n_map_ch, BUILD_MAP_RESOLUTION, BUILD_MAP_RESOLUTION),
                    dtype=np.float32,
                ),
                "worker_features": spaces.Box(
                    0.0,
                    1.0,
                    shape=(MAX_BUILD_WORKERS, WORKER_FEAT_DIM),
                    dtype=np.float32,
                ),
                "macro_vector": spaces.Box(
                    0.0,
                    1.0,
                    shape=(MACRO_VECTOR_DIM,),
                    dtype=np.float32,
                ),
                "command": spaces.Box(
                    0.0,
                    1.0,
                    shape=(COMMAND_VECTOR_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(
            [MAX_BUILD_WORKERS, BUILD_PLACEMENT_CELLS]
        )

    async def _setup_game_session(self):
        self._bot_cls = ZergBuildBot
        await super()._setup_game_session()

    async def _async_step(self, action):
        if self._bot is not None:
            await self._bot.on_build_substep()
            await self._bot.apply_build_action(np.asarray(action, dtype=np.int64))

        actions = getattr(self._bot, "_actions", None)
        if actions is None:
            actions = getattr(self._bot, "actions", None)
        if actions:
            await self._client.actions(actions[:])
            actions.clear()

        await self._client.step()
        response = await self._client.observation()
        from src.utils.sc2_observation_compat import game_state_from_response_observation

        state = game_state_from_response_observation(response.observation)
        from src.utils.proto_utils import ProtoWrapper

        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        return self._bot._get_step_results()

    def action_masks(self) -> np.ndarray:
        if self._bot is not None:
            return self._bot.action_masks()
        mask = np.ones(MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS, dtype=bool)
        mask[:MAX_BUILD_WORKERS] = False
        return mask

    def get_obs_focus_unit_tag(self) -> int:
        if self._bot is not None:
            return self._bot.get_obs_focus_unit_tag()
        return -1
