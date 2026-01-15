from train.envs.level_7_env import Level7Env
from src.level_drop_bot import LevelDropBot
from gymnasium import spaces
import numpy as np

class LevelDropEnv(Level7Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Resolve Action Heads from Config
        env_conf = self.training_conf.get("env", {})
        self.action_heads = env_conf.get("action_heads", ["move", "attack", "load", "unload"])
        
        # Dynamic Action Space
        grid_area = self.grid_size * self.grid_size
        self.action_space = spaces.Discrete(len(self.action_heads) * grid_area)

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
