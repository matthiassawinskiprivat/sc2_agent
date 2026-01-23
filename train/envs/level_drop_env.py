from train.envs.level_7_env import Level7Env
from src.level_drop_bot import LevelDropBot
from gymnasium import spaces
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot, Computer
import numpy as np

class LevelDropEnv(Level7Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize map_index based on worker_index to ensure initial diversity
        if hasattr(self, 'map_list') and self.map_list:
            self.map_index = self.worker_index % len(self.map_list)
            self.map_name = self.map_list[self.map_index]
        else:
            self.map_index = 0

    def reset(self, seed=None, options=None):
        """
        Overridden reset to rotate maps on EVERY reset call.
        """
        # If this is NOT the very first reset, rotate the map
        if self.episode_counter > 0:
            if hasattr(self, 'map_list') and self.map_list:
                self.map_index = (self.map_index + 1) % len(self.map_list)
                self.map_name = self.map_list[self.map_index]
                self._log(f"Rotating to next map: {self.map_name} (Index: {self.map_index})")
                
                # Force a hard reset to load the new map
                self.close()
        
        return super().reset(seed=seed, options=options)

    def _parse_races_from_map_name(self, map_name: str) -> tuple[Race, Race]:
        """
        Parses agent and enemy races from map name (e.g., ...Terran_vs_Zerg).
        Returns (agent_race, enemy_race).
        """
        # Default fallback
        agent_race = Race.Terran
        enemy_race = Race.Terran

        if "_vs_" in map_name:
            try:
                # 01_32x32_Lower_Left_Terran_vs_Terran -> ['...Lower_Left_Terran', 'Terran']
                parts = map_name.split("_vs_")
                agent_part = parts[0].split("_")[-1] # Take the last word before _vs_
                enemy_part = parts[1].split("_")[0] # Take the first word after _vs_

                mapping = {
                    "Terran": Race.Terran,
                    "Zerg": Race.Zerg,
                    "Protoss": Race.Protoss
                }

                agent_race = mapping.get(agent_part, Race.Terran)
                enemy_race = mapping.get(enemy_part, Race.Terran)
            except Exception as e:
                self._log(f"Error parsing races from map name {map_name}: {e}")

        return agent_race, enemy_race

    async def _setup_game_session(self):
        # Verify connection
        try:
             await self._client.ping()
        except Exception as e:
             self._log(f"Initial ping failed: {e}")
             raise e

        # 1. Create Bot instance if not exists
        if self._bot is None:
            self._log("Creating LevelDropBot...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            
            self._bot = LevelDropBot(conn=DummyConn(), training_conf=self.training_conf)
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        # 2. Parse Races and Create/Join Game
        # This preserves the pre-placed units on the map by using LevelDropBot's skip-cleanup logic
        agent_race, enemy_race = self._parse_races_from_map_name(self.map_name)
        game_map = self._resolve_map(self.map_name)

        self._log(f"Creating game on {self.map_name} ({agent_race.name} vs {enemy_race.name})...")
        await self._controller.create_game(
            game_map,
            [Bot(agent_race, self._bot), Computer(enemy_race, Difficulty.Easy)],
            realtime=self.visualize,
        )

        self._log(f"Joining game as {agent_race.name}...")
        await self._client.join_game(race=agent_race)
        
        # 3. Setup Bot internal state
        await self._client.debug_show_map()
        
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
        
        from sc2.game_state import GameState
        state = GameState(initial_observation.observation)
        
        from train.envs.level_0_env import ProtoWrapper
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

        try:
            await self._bot.on_step(0)
        except Exception:
            pass

    async def _async_step(self, action):
        """
        Specialized step logic for Level Drop: Total Annihilation win condition.
        Overrides Level7Env's combat-centric termination.
        """
        if not hasattr(self, "active_unit_tags") or not self.active_unit_tags:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_units()]
        
        if not self.active_unit_tags:
             return np.zeros(self.observation_space.shape), 0, True, False, {}

        current_tag = self.active_unit_tags[self.current_unit_index]
        
        # 1. Apply the agent's action
        await self._bot.apply_action(current_tag, action)
        
        # Send queued actions to the client before stepping
        actions = getattr(self._bot, "_actions", None)
        if actions is None:
            actions = getattr(self._bot, "actions", None)
        
        if actions:
            await self._client.actions(actions[:])
            actions.clear()
        
        # 2. Advance game physics immediately
        await self._client.step()
        
        # 3. Refresh bot state from new observation
        response = await self._client.observation()
        from sc2.game_state import GameState
        state = GameState(response.observation)
        from train.envs.level_0_env import ProtoWrapper
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

        # 4. Calculate Rewards
        combat_reward = self._bot.calculate_combat_reward()
        dist_reward = self._bot.calculate_distance_reward(current_tag)
        attrition_penalty = self._bot.calculate_allied_attrition_penalty()
        drop_reward = self._bot.calculate_drop_reward(current_tag)
        
        rewards_conf = self.training_conf.get('rewards', {})
        step_penalty = rewards_conf.get('step_penalty', -0.05)
        
        total_reward = combat_reward + dist_reward + attrition_penalty + drop_reward + step_penalty
        
        # 5. Check Win/Loss Conditions (Townhall Destruction OR Total Annihilation)
        enemy_units_count = self._bot.enemy_units.amount
        enemy_structs_count = self._bot.enemy_structures.amount
        
        townhall_types = {
            UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
            UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
            UnitTypeId.NEXUS
        }
        # Check if any enemy townhall exists
        enemy_townhalls = self._bot.enemy_structures.filter(lambda u: u.type_id in townhall_types)
        
        # Defeat: 0 controlled units (Marines + Medivacs)
        controlled_units = self._bot.get_active_units()
        allied_count = len(controlled_units)
        
        done = False
        
        # Victory Condition 1: Primary Objective (Main Structure) Destroyed
        # We assume the map started with at least one townhall.
        # If we can't find any now, we win.
        if enemy_townhalls.amount == 0 and enemy_structs_count > 0:
             # Distinguish from total annihilation for logging clarity
             done = True
             win_reward = rewards_conf.get('win_reward', 50.0)
             total_reward += win_reward
             self._log(f"VICTORY: Enemy Main Structure destroyed! Reward: {win_reward}")
             
        # Victory Condition 2: Total Annihilation (Cleanup)
        elif enemy_units_count == 0 and enemy_structs_count == 0:
            done = True
            win_reward = rewards_conf.get('win_reward', 50.0)
            total_reward += win_reward
            self._log(f"VICTORY: All enemy entities destroyed! Reward: {win_reward}")
            
        elif allied_count == 0:
            done = True
            loss_penalty = rewards_conf.get('loss_penalty', -10.0)
            total_reward += loss_penalty
            self._log(f"DEFEAT: All controlled units lost! Penalty: {loss_penalty}")
            
        # 6. Handle Cycle Advancement and Unit Deaths
        self.current_unit_index += 1
        all_active_tags = [u.tag for u in controlled_units]
        
        if not all_active_tags:
            done = True

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            self.current_unit_index = 0
            self.active_unit_tags = all_active_tags
        elif self.current_unit_index < len(self.active_unit_tags):
            # Skip any units that died in the last step
            while self.current_unit_index < len(self.active_unit_tags):
                next_tag = self.active_unit_tags[self.current_unit_index]
                if next_tag in all_active_tags:
                    break
                self.current_unit_index += 1
            
            if self.current_unit_index < len(self.active_unit_tags):
                next_tag = self.active_unit_tags[self.current_unit_index]
                obs, _ = self._bot.get_observation_for_unit(next_tag)
            else:
                # End of cycle reached early due to deaths
                self.current_unit_index = 0
                self.active_unit_tags = all_active_tags
                if self.active_unit_tags:
                    obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
                else:
                    obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
                    done = True
        else:
            # End of cycle reached normally
            self.current_unit_index = 0
            self.active_unit_tags = all_active_tags
            
            # UPDATE REWARD STATES: Only once per full cycle
            if hasattr(self._bot, "update_reward_states"):
                 self._bot.update_reward_states()
            
            if self.active_unit_tags:
                obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
                done = True

        return obs, total_reward, done, False, {}

