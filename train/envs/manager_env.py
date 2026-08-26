"""Gym environment for the learned manager agent."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import Command, CommandType
from src.agent.manager_constants import (
    DEFAULT_MANAGER_UNIT_VOCAB,
    MANAGER_COMMAND_TYPES,
    MANAGER_COUNT_BUCKETS,
    MANAGER_MACRO_VECTOR_DIM,
    MANAGER_MAP_RESOLUTION,
    MAX_MANAGER_UNIT_TYPES,
)
from src.runtime.manager_bo_penalties import evaluate_manager_emission, penalty_amount
from src.runtime.opponent_config import resolve_opponent_config
from src.transformers.manager_observation_builder import ManagerObservationBuilder
from src.utils.config_manager import ConfigManager
from src.zerg_build_bot import ZergBuildBot
from train.envs.build_env import BuildEnv


def _unit_type_from_name(name: str) -> Optional[UnitTypeId]:
    key = str(name).upper().replace(" ", "")
    if key in {"NOOP", "NONE", ""}:
        return None
    try:
        return UnitTypeId[key]
    except KeyError:
        return None


class ManagerEnv(BuildEnv):
    """
    Manager gym loop: emit Command → frozen build substeps / stub combat → rewards.

    Uses cumulative ``active_layers`` for the global map obs (unit-control contract).
    """

    def __init__(self, *args, training_conf: Dict | None = None, **kwargs):
        training_conf = dict(training_conf or {})
        level_name = kwargs.get("level_name", "level_manager_zerg_0")
        self._manager_pre_init(training_conf, level_name)
        # Skip BuildEnv.__init__ layering by calling Level0Env via BuildEnv carefully:
        # BuildEnv._build_pre_init would overwrite manager fields — so call BuildEnv
        # after we set manager fields, then redefine spaces.
        super().__init__(*args, training_conf=training_conf, **kwargs)
        self.agent_architecture = "manager"
        self.training_conf["agent_architecture"] = "manager"
        self.training_conf["keep_enemy_alive"] = True
        self._bot_cls = ZergBuildBot
        self._frozen_build_model = None
        self._manager_obs_builder = ManagerObservationBuilder(self.training_conf)
        self._define_manager_spaces()
        self._last_manager_penalty = 0.0

    def _manager_pre_init(self, training_conf: Dict, level_name: str) -> None:
        config_dir = training_conf.get("config_dir", "config_build_zerg")
        cm = ConfigManager(config_dir)
        level_conf = cm.get_level_config(level_name)
        for section in ("training", "env"):
            if section in level_conf:
                for k, v in level_conf[section].items():
                    training_conf[k] = v
        training_conf["active_layers"] = cm.get_cumulative_active_layers(level_name)
        training_conf["build_map_layers"] = cm.get_cumulative_build_map_layers(level_name)
        training_conf["build_commands"] = cm.get_cumulative_build_commands(level_name)
        training_conf["config_dir"] = config_dir
        training_conf["level_name"] = level_name
        training_conf["race"] = training_conf.get("race", "zerg")
        training_conf["keep_enemy_alive"] = True
        training_conf["use_computer"] = training_conf.get("use_computer", True)
        opp = resolve_opponent_config(training_conf)
        training_conf["opponent"] = opp
        if "difficulty" not in training_conf:
            training_conf["difficulty"] = opp.get("difficulty", "Easy")
        env_section = training_conf.setdefault("env", {})
        if isinstance(env_section, dict):
            env_section.setdefault("maps_dir", "train/maps/ladder/build_agent")
            env_section["keep_enemy_alive"] = True
            env_section.setdefault("use_computer", True)
            env_section.setdefault("difficulty", training_conf["difficulty"])
        if "rewards" not in training_conf and "env" in level_conf:
            training_conf["rewards"] = level_conf["env"].get("rewards", {})

        vocab = training_conf.get("manager_unit_vocab") or list(
            training_conf.get("build_commands") or DEFAULT_MANAGER_UNIT_VOCAB
        )
        # Ensure noop pad slot
        names = [str(x).lower() for x in vocab]
        if "noop" not in names:
            names.append("noop")
        while len(names) < MAX_MANAGER_UNIT_TYPES:
            names.append(f"_unused_{len(names)}")
        training_conf["manager_unit_vocab"] = names[:MAX_MANAGER_UNIT_TYPES]

    def _define_manager_spaces(self) -> None:
        n_ch = len(self.training_conf.get("active_layers") or []) or 1
        res = int(self.training_conf.get("manager_map_resolution", MANAGER_MAP_RESOLUTION))
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    0.0, 1.0, shape=(n_ch, res, res), dtype=np.float32
                ),
                "macro_vector": spaces.Box(
                    0.0,
                    1.0,
                    shape=(MANAGER_MACRO_VECTOR_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        n_cmds = len(MANAGER_COMMAND_TYPES)
        n_units = MAX_MANAGER_UNIT_TYPES
        n_spatial = res * res
        self.action_space = spaces.MultiDiscrete(
            [n_cmds, n_units, n_spatial, MANAGER_COUNT_BUCKETS]
        )

    def _build_pre_init(self, training_conf: Dict, level_name: str) -> None:
        # Already applied in _manager_pre_init; keep build fields intact.
        if "build_commands" not in training_conf:
            super()._build_pre_init(training_conf, level_name)

    async def _setup_game_session(self):
        self._bot_cls = ZergBuildBot
        # Manager levels: no scripted script — empty manager_script so bot waits
        # for learned emissions. Preserve YAML script if present for hybrid tests.
        await super()._setup_game_session()
        self._maybe_load_frozen_build()

    def _maybe_load_frozen_build(self) -> None:
        path = self.training_conf.get("frozen_build_checkpoint") or (
            self.training_conf.get("env") or {}
        ).get("frozen_build_checkpoint")
        if not path:
            return
        try:
            from src.agent.maskable_ppo import MaskablePPO
            from src.utils.model_utils import load_build_checkpoint

            model = MaskablePPO.load(path, device="cpu")
            load_build_checkpoint  # imported for symmetry / future graft helpers
            model.policy.set_training_mode(False)
            self._frozen_build_model = model
        except Exception:
            self._frozen_build_model = None

    def _structure_counts(self) -> Dict[UnitTypeId, int]:
        bot = self._bot
        counts: Dict[UnitTypeId, int] = {}
        if bot is None:
            return counts
        for ut in (
            UnitTypeId.SPAWNINGPOOL,
            UnitTypeId.ROACHWARREN,
            UnitTypeId.EXTRACTOR,
            UnitTypeId.HATCHERY,
        ):
            try:
                counts[ut] = int(bot.structures(ut).amount)
            except Exception:
                counts[ut] = 0
        return counts

    def _has_pool(self) -> bool:
        return self._structure_counts().get(UnitTypeId.SPAWNINGPOOL, 0) > 0

    def _decode_manager_action(self, action: np.ndarray) -> Command:
        cmd_i = int(action[0]) % len(MANAGER_COMMAND_TYPES)
        unit_i = int(action[1]) % MAX_MANAGER_UNIT_TYPES
        spatial_i = int(action[2])
        count_i = int(action[3]) % MANAGER_COUNT_BUCKETS
        cmd_type = MANAGER_COMMAND_TYPES[cmd_i]
        vocab = self.training_conf.get("manager_unit_vocab") or list(DEFAULT_MANAGER_UNIT_VOCAB)
        unit_name = vocab[unit_i] if unit_i < len(vocab) else "noop"
        unit_type = _unit_type_from_name(unit_name)
        res = int(self.training_conf.get("manager_map_resolution", MANAGER_MAP_RESOLUTION))
        gx = spatial_i % res
        gy = spatial_i // res
        # Coarse sector → playable-relative world point when bot is ready.
        target = Point2((float(gx), float(gy)))
        bot = self._bot
        if bot is not None and getattr(bot, "game_info", None) is not None:
            playable = bot.game_info.playable_area
            w = float(getattr(playable, "width", res) or res)
            h = float(getattr(playable, "height", res) or res)
            ox = float(getattr(playable, "x", 0) or 0)
            oy = float(getattr(playable, "y", 0) or 0)
            target = Point2(
                (
                    ox + (gx + 0.5) * (w / float(res)),
                    oy + (gy + 0.5) * (h / float(res)),
                )
            )
        count = max(1, count_i) if count_i > 0 else 1
        worker_count = count_i if count_i > 0 else None
        return Command(
            command_type=cmd_type,
            unit_type_id=unit_type,
            target_pos=target,
            count=count if cmd_type == CommandType.TRAIN else None,
            worker_count=worker_count
            if cmd_type in {CommandType.MINE, CommandType.MINE_GAS}
            else None,
        )

    def _manager_observation(self) -> Dict[str, np.ndarray]:
        if self._bot is None:
            n_ch = len(self.training_conf.get("active_layers") or []) or 1
            res = int(self.training_conf.get("manager_map_resolution", MANAGER_MAP_RESOLUTION))
            return {
                "map": np.zeros((n_ch, res, res), dtype=np.float32),
                "macro_vector": np.zeros(MANAGER_MACRO_VECTOR_DIM, dtype=np.float32),
            }
        return self._manager_obs_builder.build(self._bot)

    async def _run_build_substeps(self, n: int = 8) -> float:
        """Advance physics with frozen (or idle) build policy; return build reward sum."""
        total = 0.0
        for _ in range(max(1, n)):
            if self._bot is None:
                break
            await self._bot.on_build_substep()
            build_action = self._sample_build_action()
            await self._bot.apply_build_action(np.asarray(build_action, dtype=np.int64))
            actions = getattr(self._bot, "_actions", None) or getattr(self._bot, "actions", None)
            if actions:
                await self._client.actions(actions[:])
                actions.clear()
            await self._client.step()
            response = await self._client.observation()
            from src.utils.sc2_observation_compat import game_state_from_response_observation
            from src.utils.proto_utils import ProtoWrapper

            state = game_state_from_response_observation(response.observation)
            self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
            _obs, reward, done, _trunc, _info = self._bot._get_step_results()
            total += float(reward)
            if done:
                break
            if self._bot.command_queue.active is None:
                break
        return total

    def _sample_build_action(self) -> np.ndarray:
        from src.agent.build_constants import BUILD_PLACEMENT_CELLS, MAX_BUILD_WORKERS

        if self._frozen_build_model is not None and self._bot is not None:
            try:
                obs, _ = self._bot._get_observation()
                action, _ = self._frozen_build_model.predict(obs, deterministic=True)
                return np.asarray(action, dtype=np.int64)
            except Exception:
                pass
        return np.array([0, BUILD_PLACEMENT_CELLS // 2], dtype=np.int64)

    async def _async_step(self, action):
        self._last_manager_penalty = 0.0
        cmd = self._decode_manager_action(np.asarray(action, dtype=np.int64))
        reward = float(self.training_conf.get("rewards", {}).get("step_penalty", -0.02))

        if cmd.command_type != CommandType.NO_OP and self._bot is not None:
            violation = evaluate_manager_emission(
                cmd,
                has_pool=self._has_pool(),
                structure_counts=self._structure_counts(),
            )
            if violation:
                pen = penalty_amount(self.training_conf.get("rewards", {}), violation)
                reward += pen
                self._last_manager_penalty = pen
            self._bot.command_queue.enqueue(cmd)
            self._bot.command_queue.activate_next()

        substeps = int(self.training_conf.get("macro_tick_interval", 8))
        reward += await self._run_build_substeps(substeps)

        # Outcome vs computer
        done = False
        truncated = False
        if self._bot is not None:
            if not self._bot.townhalls or (
                hasattr(self._bot.townhalls, "exists") and not self._bot.townhalls.exists
            ):
                reward += float(self.training_conf.get("rewards", {}).get("loss", -100.0))
                done = True
            enemy_th = getattr(self._bot, "enemy_structures", None)
            if enemy_th is not None:
                try:
                    enemy_hatch = enemy_th(UnitTypeId.HATCHERY)
                    if hasattr(enemy_hatch, "exists") and not enemy_hatch.exists:
                        # Also check lair/hive
                        lair = enemy_th(UnitTypeId.LAIR)
                        hive = enemy_th(UnitTypeId.HIVE)
                        no_lair = not getattr(lair, "exists", False)
                        no_hive = not getattr(hive, "exists", False)
                        if no_lair and no_hive:
                            reward += float(
                                self.training_conf.get("rewards", {}).get("win", 100.0)
                            )
                            done = True
                except Exception:
                    pass
            self.steps = getattr(self, "steps", 0) + 1
            max_steps = int(self.training_conf.get("max_steps", 2000))
            if self.steps >= max_steps:
                truncated = True

        obs = self._manager_observation()
        info = {"manager_penalty": self._last_manager_penalty}
        return obs, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        """Pad-only masks: unused unit vocab slots; no strategic BO masking."""
        n_cmds = len(MANAGER_COMMAND_TYPES)
        n_units = MAX_MANAGER_UNIT_TYPES
        res = int(self.training_conf.get("manager_map_resolution", MANAGER_MAP_RESOLUTION))
        n_spatial = res * res
        n_count = MANAGER_COUNT_BUCKETS
        mask = np.ones(n_cmds + n_units + n_spatial + n_count, dtype=bool)
        vocab = self.training_conf.get("manager_unit_vocab") or []
        for i, name in enumerate(vocab):
            if str(name).startswith("_unused"):
                mask[n_cmds + i] = False
        return mask

    async def _async_reset(self, seed=None, options=None):
        result = await super()._async_reset(seed=seed, options=options)
        self.steps = 0
        # BuildEnv returns build obs; replace with manager obs.
        if isinstance(result, tuple) and len(result) == 2:
            return self._manager_observation(), result[1]
        return self._manager_observation()
