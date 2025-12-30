from train.envs.level_0_env import Level0Env, ProtoWrapper
from src.level_3_bot import Level3Bot
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
import numpy as np

class Level3Env(Level0Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load specific Level 3 configuration
        # Level0Env initialized config_manager
        level_conf = self.config_manager.get_level_config(self.level_name)
        self.targets = level_conf.get("env", {}).get("targets", 1)
        self.current_unit_index = 0
        self.active_unit_tags = []
        
    def _log(self, message):
        # Prefix logs with Level3Env to distinguish
        super()._log(f"[Level3Env] {message}")

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
            # Use Level3Bot instead of SC2Bot
            self._bot = Level3Bot(conn=DummyConn(), training_conf=self.training_conf)
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        self._log("Creating game...")
        await self._controller.create_game(
            game_map,
            [Bot(Race.Zerg, self._bot), Computer(Race.Zerg, Difficulty.Easy)],
            realtime=self.visualize,
        )
        await self._client.join_game(race=Race.Zerg)
        self._bot._generated_frame = -1
        game_info = await self._client.get_game_info()
        game_data = await self._client.get_game_data()

        self._log("Preparing bot start...")
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
            
        self._bot._prepare_step(state, ProtoWrapper(game_info._proto))
        try:
            await self._bot.on_step(0)
        except Exception as e:
            self._log(f"Error in on_step(0): {e}")
            raise

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
             
        return obs, info

    async def _async_step(self, action):
        # Identify current unit
        if not hasattr(self, "active_unit_tags") or not self.active_unit_tags:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_zerglings()]
        
        if not self.active_unit_tags:
             # No units? Done.
             return np.zeros(self.observation_space.shape), 0, True, False, {}
             
        current_tag = self.active_unit_tags[self.current_unit_index]
        
        # Apply action
        await self._bot.apply_action(current_tag, action)
        
        # Advance index
        self.current_unit_index += 1
        
        if self.current_unit_index < len(self.active_unit_tags):
            # Next unit in cycle
            next_tag = self.active_unit_tags[self.current_unit_index]
            obs, _ = self._bot.get_observation_for_unit(next_tag)
            return obs, 0.0, False, False, {}
        else:
            # End of cycle. Step physics.
            # Send batched actions (SC2Bot actions list is populated by apply_action via do())
            actions = getattr(self._bot, "actions", [])
            if actions:
                await self._client.actions(actions)
                self._bot.actions.clear()

            await self._client.step()
            
            # Prepare bot state for next step (update units etc)
            response = await self._client.observation()
            from sc2.game_state import GameState
            state = GameState(response.observation)
            self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

            # Check collections
            collection_reward = await self._bot.check_collections()
            
            # Calculate distance reward
            dist_reward = self._calculate_avg_distance_reward()
            
            total_reward = collection_reward + dist_reward
            
            # Update active tags for next cycle (remove dead ones)
            self.active_unit_tags = [u.tag for u in self._bot.get_active_zerglings()]
            
            # Done condition
            beacons_left = self._bot.units(UnitTypeId.BEACON_TERRANSMALL).amount
            zerglings_left = self._bot.units(UnitTypeId.ZERGLING).amount
            
            done = False
            if beacons_left == 0:
                done = True # Success
            elif zerglings_left == 0:
                done = True # Failure
            
            # Prepare next obs (Unit 0)
            self.current_unit_index = 0
            if self.active_unit_tags:
                obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            
            return obs, total_reward, done, False, {}

    def _calculate_avg_distance_reward(self):
        zerglings = self._bot.units(UnitTypeId.ZERGLING)
        beacons = self._bot.units(UnitTypeId.BEACON_TERRANSMALL)
        
        if not zerglings or not beacons:
            return 0.0
            
        rewards_conf = self.training_conf.get('rewards', {})
        step_penalty = rewards_conf.get('step_penalty', -0.05)
        dist_multiplier = rewards_conf.get('distance_multiplier', 0.1)
        
        total_dist_change = 0.0
        
        current_distances = {}
        for z in zerglings:
            # Find nearest beacon
            nearest_beacon = beacons.closest_to(z)
            dist = z.distance_to(nearest_beacon)
            current_distances[z.tag] = dist
            
            prev_dist = self._bot.previous_distances.get(z.tag)
            if prev_dist is not None:
                change = prev_dist - dist
                total_dist_change += change
        
        # Update previous
        self._bot.previous_distances = current_distances
        
        # Average distance reward
        avg_dist_reward = (total_dist_change * dist_multiplier) / max(1, len(zerglings))
        
        return avg_dist_reward + step_penalty


