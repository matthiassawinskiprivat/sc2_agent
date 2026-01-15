from train.envs.level_7_env import Level7Env
from src.level_drop_bot import LevelDropBot
from gymnasium import spaces
from sc2.data import Race
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

        # 2. Join the existing game instead of creating a new one
        # This preserves the pre-placed units on the map
        self._log(f"Joining existing game on {self.map_name}...")
        # Since maps are pre-configured with players, we just join.
        # We assume Player 1 is the agent.
        await self._client.join_game(race=Race.Terran)
        
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
        # We need to intercept the reward calculation
        # Call super step (which does bot.apply_action + client.step)
        obs, reward, done, truncated, info = await super()._async_step(action)
        
        # Add Drop Rewards
        drop_reward = self._bot.calculate_drop_reward()
        reward += drop_reward
        
        return obs, reward, done, truncated, info
