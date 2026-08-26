import numpy as np
from gymnasium import spaces as gym_spaces
from sc2.data import Race, Difficulty
from sc2.player import Bot, Computer

from src.level_3_bot import Level3Bot
from train.envs.level_0_env import Level0Env, ProtoWrapper
from train.envs.sc2_session_config import (
    DEFAULT_POST_CREATE_GAME_DELAY_SECS,
    host_local_sc2_game,
    session_realtime,
    use_computer_player,
)


class Level3Env(Level0Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load specific Level 3 configuration
        # Level0Env initialized config_manager
        level_conf = self.config_manager.get_level_config(self.level_name)
        env_conf = level_conf.get("env", {})
        
        # Priority: explicit training_conf['env'] > level_conf['env']
        merged_env_conf = self.training_conf.get("env", {}).copy()
        for k, v in env_conf.items():
            if k not in merged_env_conf:
                merged_env_conf[k] = v

        self.targets = merged_env_conf.get("targets", 1)
        
        # Ensure targets and other env-level configs are propagated to training_conf 
        # so they reach the Level3Bot
        self.training_conf["targets"] = self.targets
        if "rewards" in merged_env_conf:
            # Merge rewards too
            r = self.training_conf.get("rewards", {}).copy()
            r.update(merged_env_conf["rewards"])
            self.training_conf["rewards"] = r
            
        self.current_unit_index = 0
        self.active_unit_tags = []
        
    def _log(self, message):
        # Prefix logs with Level3Env to distinguish
        super()._log(f"[Level3Env] {message}")

    def get_obs_focus_unit_tag(self) -> int:
        """Tag of the unit whose egocentric observation is returned on the next step (see Level0Env)."""
        tags = getattr(self, "active_unit_tags", None) or []
        idx = getattr(self, "current_unit_index", 0)
        if not tags or idx >= len(tags):
            return -1
        return int(tags[idx])

    async def _async_preview_tick(self):
        stepped = await self._async_apply_preview_spatial_moves()
        if not stepped:
            await self._client.step()
        response = await self._client.observation()
        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(response.observation)
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        if not getattr(self, "active_unit_tags", None):
            self.active_unit_tags = [u.tag for u in self._bot.get_active_zerglings()]
        if self.active_unit_tags:
            tag = self.active_unit_tags[self.current_unit_index % len(self.active_unit_tags)]
            obs, _ = self._bot.get_observation_for_unit(tag)
            return obs
        obs, _, _, _, _ = self._bot._get_step_results()
        return obs

    def action_masks(self):
        if self._bot:
            # Robust unit identification for masking
            if hasattr(self, "active_unit_tags") and 0 <= self.current_unit_index < len(self.active_unit_tags):
                self._bot.active_unit_tag = self.active_unit_tags[self.current_unit_index]
            else:
                self._bot.active_unit_tag = None
            self._bot.current_unit_index = self.current_unit_index
        return super().action_masks()

    async def _setup_game_session(self):
        # Verify connection
        try:
             await self._client.ping()
        except Exception as e:
             self._log(f"Initial ping failed: {e}")
             raise e

        game_map = self._resolve_map(self.map_name)
        if self._bot is None:
            self._log("Creating Level3Bot...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            
            if self.agent_architecture == "universal":
                from src.sc2_bot_universal import SC2BotUniversal
                # Inherit from Level3Bot only, but delegate key universal methods
                class UniversalLevel3Bot(Level3Bot):
                    def __init__(self, conn, training_conf=None):
                        super().__init__(conn, training_conf)
                        # Manually trigger SC2BotUniversal's logic if needed
                        # But SC2BotUniversal.__init__ just sets up agent_system
                        # which is already handled by SC2Bot base class.
                    
                    def action_masks(self, curriculum_progress=1.0):
                        return SC2BotUniversal.action_masks(self, curriculum_progress)

                    async def apply_action(self, unit_tag, action):
                        return await SC2BotUniversal.apply_action(self, unit_tag, action)

                    def _get_zero_observation(self):
                        return SC2BotUniversal._get_zero_observation(self)

                    def _get_observation(self, active_unit=None):
                        # Use Level3Bot's logic but it will call its own _get_observation
                        # which we need to make sure uses the universal processing
                        return Level3Bot._get_observation(self, active_unit=active_unit)
                
                self._bot = UniversalLevel3Bot(conn=DummyConn(), training_conf=self.training_conf)
            else:
                self._bot = Level3Bot(conn=DummyConn(), training_conf=self.training_conf)
                
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        # Match Level0Env: movement curriculum maps must NOT add a Computer opponent.
        # A Zerg AI will spawn hatchery/larva/overlords etc.; cleanup cannot clear the map
        # and MapCleanupFailedError will fire on every reset including fresh SC2 launches.
        self._log("Creating game...")
        use_computer = use_computer_player(self.level_name, self.training_conf)

        players = [Bot(Race.Zerg, self._bot)]
        if use_computer:
            difficulty_name = self.training_conf.get("env", {}).get("difficulty", "Easy")
            try:
                difficulty = getattr(Difficulty, difficulty_name)
            except AttributeError:
                difficulty = Difficulty.Easy
            players.append(Computer(Race.Zerg, difficulty))
            self._log(f"Starting game with Computer ({difficulty_name})")
        else:
            self._log("Starting game without Computer (solo Bot)")

        sim_realtime = session_realtime(self.training_conf)
        post_delay = float(
            self.training_conf.get(
                "post_create_game_delay_sec",
                DEFAULT_POST_CREATE_GAME_DELAY_SECS,
            )
        )
        self._client = await host_local_sc2_game(
            controller=self._controller,
            client_cls=self._client_cls,
            game_map=game_map,
            players=players,
            realtime=sim_realtime,
            join_race=Race.Zerg,
            log=self._log,
            post_create_delay_sec=post_delay,
            player_name="Trainee",
        )

        self._bot._generated_frame = -1
        game_info = await self._client.get_game_info()
        game_data = await self._client.get_game_data()

        self._log("Preparing bot start...")
        self._bot.client = self._client
        try:
            from sc2.unit import UnitGameData
            UnitGameData._game_data = game_data
            UnitGameData._bot_object = self._bot
            self._bot._game_info = game_info
            self._bot._game_data = game_data
        except (ImportError, AttributeError):
            pass

        self._bot._prepare_start(
            client=self._client,
            player_id=1,
            game_info=game_info,
            game_data=game_data,
            realtime=sim_realtime,
        )
        initial_observation = await self._client.observation()

        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(initial_observation.observation)
            
        self._bot._prepare_step(state, ProtoWrapper(game_info._proto))
        try:
            await self._bot.on_step(0)
        except Exception as e:
            self._log(f"Error in on_step(0): {e}")
            raise

    def _get_zero_observation(self):
        """Returns a properly shaped zero observation matching the space (Dict or Discrete)."""
        if isinstance(self.observation_space, gym_spaces.Dict):
             return {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.observation_space.spaces.items()}
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def reset(self, seed=None, options=None):
        self.current_unit_index = 0
        obs, info = super().reset(seed, options)
        
        if self._bot:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_zerglings()]
             if self.active_unit_tags:
                 # Override observation to focus on first unit
                 obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
        else:
             self.active_unit_tags = []
             # Ensure obs matches space even if bot setup fails early
             if not isinstance(obs, dict) and isinstance(self.observation_space, gym_spaces.Dict):
                 obs = self._get_zero_observation()
             
        return obs, info

    async def _async_step(self, action):
        """
        Modified step function for Level 3: Individual Unit Attribution.
        Each unit action now triggers a game step and returns its own reward.
        This ensures perfect credit assignment for each unit's behavior.
        """
        # Identify current active units if not already cached
        if not hasattr(self, "active_unit_tags") or not self.active_unit_tags:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_zerglings()]
        
        if not self.active_unit_tags:
             # No controllable units left:
             # - Success only if all beacons are already gone.
             # - Otherwise terminate as truncated (unsuccessful ending).
             beacons_left = self._bot.all_units(self._bot.beacon_type).amount
             done = (beacons_left == 0)
             truncated = (not done)
             return self._get_zero_observation(), 0, done, truncated, {}
             
        current_tag = self.active_unit_tags[self.current_unit_index]
        
        # 1. Apply the agent's action to the current unit
        await self._bot.apply_action(current_tag, action)
        
        # Send queued actions to the client before stepping
        actions = getattr(self._bot, "_actions", None)
        if actions is None:
            actions = getattr(self._bot, "actions", None)
        
        if actions:
            await self._client.actions(actions[:])
            actions.clear()
        
        # 2. Advance game physics immediately for this unit's action
        # This allows immediate individual reward calculation
        await self._client.step()
        
        # 3. Refresh bot state from the new observation
        response = await self._client.observation()
        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(response.observation)
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

        # 4. Calculate Individual Reward for THIS unit
        # Distance Shaping (Progress towards assigned beacon)
        dist_reward = self._bot.calculate_distance_reward(current_tag)
        
        # Collection Reward (Did ANY unit reach a beacon?)
        # check_collections now buffers rewards in self._bot.pending_success_rewards
        await self._bot.check_collections()
        
        # Attribution: Retrieve THIS unit's buffered success reward
        success_reward = self._bot.pending_success_rewards.pop(current_tag, 0.0)
        
        # If this unit just got its success reward, finalize its removal from the game
        if success_reward > 0:
            # Kill unit and its assigned beacon
            b_tag = self._bot.current_assignments.get(current_tag)
            tags_to_kill = [current_tag]
            if b_tag:
                tags_to_kill.append(b_tag)
            
            await self._client.debug_kill_unit(tags_to_kill)
            
            # Clean up assignment for THIS unit only
            if current_tag in self._bot.current_assignments:
                del self._bot.current_assignments[current_tag]
            
            # Remaining units keep their episode-initial Hungarian pairing (no global re-match).
            
            if current_tag in self._bot.completed_tags:
                self._bot.completed_tags.remove(current_tag)
            if b_tag in self._bot.completed_tags:
                self._bot.completed_tags.remove(b_tag)

        # Step Penalty (Per action to encourage speed)
        rewards_conf = self.training_conf.get('rewards', {})
        step_penalty = rewards_conf.get('step_penalty', -0.05)
        
        total_reward = dist_reward + success_reward + step_penalty
        
        # 5. Handle Cycle Advancement and Unit Deaths
        self.current_unit_index += 1
        all_active_tags = [u.tag for u in self._bot.get_active_zerglings()]
        
        # Termination conditions — must match Level3Bot beacon visibility (neutral beacons live in
        # `all_units`, never in `self._bot.units`). Falling back to `units` makes beacons_left==0
        # every step → done every step → constant VecEnv resets and near-zero GPU/CPU utilization.
        beacons_left = self._bot.all_units(self._bot.beacon_type).amount
        zerglings_left = len(all_active_tags)
        
        # Gymnasium semantics for Level 3:
        # - done=True only when all targets are collected.
        # - truncated=True for unsuccessful terminal endings (e.g. no units left).
        # Time-limit truncation is also applied by Level0Env.step() when max_steps is reached.
        done = (beacons_left == 0)
        truncated = False
        if (not done) and (zerglings_left == 0):
            truncated = True
            
        # Determine the next observation
        if done:
            obs = self._get_zero_observation()
            self.current_unit_index = 0
            self.active_unit_tags = all_active_tags
        elif self.current_unit_index < len(self.active_unit_tags):
            # Still in the middle of the original cycle. 
            # Skip any units that might have died or been collected.
            while self.current_unit_index < len(self.active_unit_tags):
                next_tag = self.active_unit_tags[self.current_unit_index]
                if next_tag in all_active_tags:
                    break
                self.current_unit_index += 1
            
            if self.current_unit_index < len(self.active_unit_tags):
                next_tag = self.active_unit_tags[self.current_unit_index]
                obs, _ = self._bot.get_observation_for_unit(next_tag)
            else:
                # All remaining units in cycle are gone, restart cycle
                self.current_unit_index = 0
                self.active_unit_tags = all_active_tags
                if self.active_unit_tags:
                    obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
                else:
                    obs = self._get_zero_observation()
                    truncated = True
        else:
            # End of cycle reached normally
            self.current_unit_index = 0
            self.active_unit_tags = all_active_tags
            if self.active_unit_tags:
                obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
            else:
                obs = self._get_zero_observation()
                truncated = True

        return obs, total_reward, done, truncated, {}




