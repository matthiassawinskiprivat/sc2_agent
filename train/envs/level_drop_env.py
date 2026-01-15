from train.envs.level_7_env import Level7Env
from src.level_drop_bot import LevelDropBot
from gymnasium import spaces
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

    def _setup_game_session(self):
        # We need to inject LevelDropBot instead of Level7Bot
        if self._bot is None:
            self._log("Creating LevelDropBot...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            
            self._bot = LevelDropBot(conn=DummyConn(), training_conf=self.training_conf)
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size
            
        # Call grandparent setup to avoid Level7Env creating Level7Bot
        # But Level7Env._setup_game_session is monolithic.
        # We must rely on self._bot being set BEFORE calling super().
        # Level7Env check: "if self._bot is None:" -> We are safe.
        
        return super()._setup_game_session()

    async def _async_step(self, action):
        # We need to intercept the reward calculation
        # Call super step (which does bot.apply_action + client.step)
        obs, reward, done, truncated, info = await super()._async_step(action)
        
        # Add Drop Rewards
        drop_reward = self._bot.calculate_drop_reward()
        reward += drop_reward
        
        return obs, reward, done, truncated, info
