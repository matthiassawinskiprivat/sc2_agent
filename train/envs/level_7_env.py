from train.envs.level_3_env import Level3Env
from src.level_7_bot import Level7Bot
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from gymnasium import spaces
import numpy as np

class Level7Env(Level3Env):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Unit and Enemy types from config — read from flat root first (set by Level0Env's
        # flattening of env params), then fall back to nested env dict for direct instantiation.
        self.unit_type = (
            self.training_conf.get("unit_type")
            or self.training_conf.get("env", {}).get("unit_type", "MARINE")
        ).upper()
        self.enemy_type = (
            self.training_conf.get("enemy_type")
            or self.training_conf.get("env", {}).get("enemy_type", "ZERGLING")
        ).upper()

    def get_race_from_unit_type(self, unit_type_name: str) -> Race:
        """Maps unit type names to SC2 Races."""
        mapping = {
            "MARINE": Race.Terran,
            "MARAUDER": Race.Terran,
            "ROACH": Race.Zerg,
            "ZERGLING": Race.Zerg,
            "HYDRALISK": Race.Zerg,
            "STALKER": Race.Protoss,
            "ZEALOT": Race.Protoss,
            "COLOSSUS": Race.Protoss,
        }
        return mapping.get(unit_type_name.upper(), Race.Terran)

    def _log(self, message):
        super()._log(f"[Level7Env] {message}")

    def action_masks(self):
        """
        Retrieves and validates the action mask shape.
        """
        if self._bot:
            if hasattr(self, "active_unit_tags") and 0 <= self.current_unit_index < len(self.active_unit_tags):
                self._bot.active_unit_tag = self.active_unit_tags[self.current_unit_index]
            else:
                self._bot.active_unit_tag = None
            self._bot.current_unit_index = self.current_unit_index
        mask = super().action_masks()
        if isinstance(self.action_space, spaces.Discrete):
            if mask.shape[0] != self.action_space.n:
                self._log(f"ERROR: Mask shape mismatch! Expected {self.action_space.n}, got {mask.shape[0]}")
                return np.ones(self.action_space.n, dtype=bool)
        return mask

    async def _setup_game_session(self):
        # Override to use Level7Bot and dynamic Race
        try:
             await self._client.ping()
        except Exception as e:
             self._log(f"Initial ping failed: {e}")
             raise e

        game_map = self._resolve_map(self.map_name)
        if self._bot is None:
            self._log("Creating Level7Bot...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            
            self._bot = Level7Bot(conn=DummyConn(), training_conf=self.training_conf)
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        allied_race = self.get_race_from_unit_type(self.unit_type)
        enemy_race = self.get_race_from_unit_type(self.enemy_type)

        self._log(f"Creating game ({allied_race.name} vs {enemy_race.name})...")
        await self._controller.create_game(
            game_map,
            [Bot(allied_race, self._bot), Computer(enemy_race, Difficulty.Easy)],
            realtime=self.visualize,
        )
        await self._client.join_game(race=allied_race)
        # Enable full vision for unit cleanup and observation.
        # Mark the flag on the bot so reset_game_state doesn't toggle it back.
        await self._client.debug_show_map()
        self._bot._debug_show_map_enabled = True
        
        self._bot._generated_frame = -1
        game_info = await self._client.get_game_info()
        game_data = await self._client.get_game_data()

        try:
            from sc2.unit import UnitGameData
            UnitGameData._game_data = game_data
            UnitGameData._bot_object = self._bot
            self._bot.client = self._client
            self._bot._game_info = game_info
            self._bot._game_data = game_data
        except ImportError:
            pass
        
        self._bot._prepare_start(client=self._client, player_id=1, game_info=game_info, game_data=game_data)
        initial_observation = await self._client.observation()

        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(initial_observation.observation)
        
        from train.envs.level_0_env import ProtoWrapper
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

        try:
            await self._bot.on_step(0)
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        if self._bot:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_units()]
             self.current_unit_index = 0
             if self.active_unit_tags:
                 obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
        
        return obs, info

    def _has_living_combat_enemies(self) -> bool:
        """True when a combat timeout still left enemies alive."""
        if not self._bot:
            return False
        try:
            enemies = self._bot.enemy_units(self._bot.enemy_unit_type)
        except Exception:
            return False

        amount = getattr(enemies, "amount", None)
        if amount is not None:
            return int(amount) > 0
        if getattr(enemies, "exists", False):
            return True
        return False

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        # Movement levels keep Level0Env's bootstrap-on-time-limit behavior. Combat levels
        # should not learn that safely waiting out the clock is acceptable when enemies live.
        if (
            self.steps >= self.max_steps
            and truncated
            and not done
            and self._has_living_combat_enemies()
        ):
            rewards_conf = self.training_conf.get("rewards", {})
            timeout_penalty = rewards_conf.get(
                "timeout_penalty",
                rewards_conf.get("loss_penalty", -50.0),
            )
            reward += timeout_penalty
            done = True
            truncated = False
            info = dict(info or {})
            info["timeout_failure"] = True

        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        """
        Modified step logic for Level 7: Individual Unit Attribution.
        Each unit action triggers a game step to ensure precise combat reward attribution.
        """
        if not hasattr(self, "active_unit_tags") or not self.active_unit_tags:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_units()]
        
        if not self.active_unit_tags:
             return self._get_zero_observation(), 0, True, False, {}

        current_tag = self.active_unit_tags[self.current_unit_index]

        # 1. Snapshot pre-action kiting state so the retreat reward compares
        #    post-step distance against the distance the agent observed (single-step
        #    delta), not against the end of this unit's previous turn which can be
        #    N game steps ago in multi-unit cycling.
        self._bot.record_pre_action_kiting_state(current_tag)

        # 2. Apply the agent's action
        # The policy is the sole source of unit orders; no bot-side auto-attack helper
        # fills in idle non-acting units. Removing the helper avoids a hidden actuation
        # path that overrode the agent's chosen kiting moves and inflated reward credit
        # for units the policy never directed (agent-faithful actuation, AGENTS.md).
        await self._bot.apply_action(current_tag, action)

        # Send queued actions to the client before stepping
        actions = getattr(self._bot, "_actions", None)
        if actions is None:
            actions = getattr(self._bot, "actions", None)
        
        if actions:
            await self._client.actions(actions[:])
            actions.clear()
        
        # 3. Advance game physics
        await self._client.step()
        
        # 4. Refresh bot state from new observation
        response = await self._client.observation()
        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(response.observation)
        from train.envs.level_0_env import ProtoWrapper
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        # Keep combat assignments synced with latest living enemies/allies.
        self._bot.refresh_combat_assignments()

        # 5. Calculate reward
        attributed_combat, _ = self._bot.calculate_unit_centric_combat_delta(current_tag)
        allied_attrition = self._bot.calculate_allied_attrition_penalty()

        # Flush any pending rewards buffered under dead ally tags (they will never
        # become current_tag again, so their credit would otherwise be silently lost).
        alive_ally_tags = {u.tag for u in self._bot.get_active_units()}
        for dead_tag in list(self._bot._pending_rewards.keys()):
            if dead_tag != current_tag and dead_tag not in alive_ally_tags:
                attributed_combat += self._bot._pending_rewards.pop(dead_tag)

        dist_reward = self._bot.calculate_distance_reward(current_tag)

        shaping_reward = self._bot.calculate_kiting_simple_shaping(current_tag)
        use_global_reward = self._bot.calculate_use_global_reward(current_tag)

        rewards_conf = self.training_conf.get('rewards', {})
        step_penalty = rewards_conf.get('step_penalty', -0.05)

        total_reward = (
            attributed_combat
            + allied_attrition
            + dist_reward
            + shaping_reward
            + use_global_reward
            + step_penalty
        )

        # Check Win/Loss Conditions
        allied_count = self._bot.units(self._bot.allied_unit_type).amount
        enemy_count = self._bot.enemy_units(self._bot.enemy_unit_type).amount

        done = False
        if enemy_count == 0:
            done = True
            total_reward += rewards_conf.get('win_reward', 50.0)
        elif allied_count == 0:
            done = True
            total_reward += rewards_conf.get('loss_penalty', -10.0)
            
        # 5. Handle Cycle Advancement and Unit Deaths
        self.current_unit_index += 1
        all_active_tags = [u.tag for u in self._bot.get_active_units()]
        
        if not all_active_tags:
            done = True

        if done:
            obs = self._get_zero_observation()
            self.current_unit_index = 0
            self.active_unit_tags = all_active_tags
        elif self.current_unit_index < len(self.active_unit_tags):
            # Skip any units that died in the last step.
            while self.current_unit_index < len(self.active_unit_tags):
                next_tag = self.active_unit_tags[self.current_unit_index]
                if next_tag in all_active_tags:
                    break
                self.current_unit_index += 1

            if self.current_unit_index < len(self.active_unit_tags):
                next_tag = self.active_unit_tags[self.current_unit_index]
                obs, _ = self._bot.get_observation_for_unit(next_tag)
            else:
                # End of cycle reached early due to deaths.
                self.current_unit_index = 0
                self.active_unit_tags = all_active_tags
                if self.active_unit_tags:
                    obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
                else:
                    obs = self._get_zero_observation()
                    done = True
        else:
            # End of cycle reached normally.
            self.current_unit_index = 0
            self.active_unit_tags = all_active_tags
            if self.active_unit_tags:
                obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
            else:
                obs = self._get_zero_observation()
                done = True

        return obs, total_reward, done, False, {}

