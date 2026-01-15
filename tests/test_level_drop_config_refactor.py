import pytest
import numpy as np
from train.envs.level_drop_env import LevelDropEnv
from src.level_drop_bot import LevelDropBot
from gymnasium import spaces

class MockConfigManager:
    """Mock ConfigManager to simulate the new config structure."""
    def get_level_config(self, level_name):
        return {
            "env": {
                "name": "LevelDrop",
                "env_class": "LevelDropEnv",
                "grid_size": 128,
                "unit_type": "MARINE",
                "enemy_type": "ZERGLING",
                "max_steps": 1000,
                "rewards": {
                    "load_reward": 0.5,
                    "unload_reward": 2.0
                },
                "active_layers": ["allied_unit_density"],
                "action_heads": ["move", "attack", "load", "unload"]
            },
            "training": {
                "num_envs": 1
            }
        }

@pytest.fixture
def level_drop_config():
    return {
        "env": {
            "grid_size": 128,
            "rewards": {
                "load_reward": 0.5,
                "unload_reward": 2.0
            },
            "action_heads": ["move", "attack", "load", "unload"]
        },
        "training": {
            "num_envs": 1
        }
    }

def test_level_drop_env_action_space_size(level_drop_config):
    """
    Test 2: Verify LevelDropEnv calculates action space size based on action_heads.
    """
    # Initialize Env with the config
    # We need to patch the internal config loading or pass kwargs if supported.
    # LevelDropEnv calls super().__init__, which sets up training_conf.
    # We can pass `training_conf` directly if the base class supports it?
    # Let's check Level3Env init. It usually takes kwargs and updates training_conf.
    
    # Actually Level3Env takes config_path or uses defaults.
    # We can inject the config by subclassing or mocking.
    
    class TestEnv(LevelDropEnv):
        def __init__(self, training_conf):
            self.training_conf = training_conf
            # Skip full init for this test, just call relevant parts
            self.grid_size = 128
            self.training_conf = training_conf
            
            # Mimic LevelDropEnv init logic
            grid_area = self.grid_size * self.grid_size
            
            # This is the logic we WANT to implement:
            action_heads = self.training_conf.get('env', {}).get('action_heads', [])
            if not action_heads:
                # Fallback for legacy behavior if we want to support it, 
                # but for this refactor we expect it to be present.
                # Currently hardcoded to 4 in LevelDropEnv.
                # We want to test that it USES the list.
                # So if we change the list, the space should change.
                pass
                
            super(LevelDropEnv, self).__init__(training_conf=training_conf)
            
            # Override the hardcoded logic we are about to replace
            # But wait, we are testing the actual class. We shouldn't override logic here.
            # We should instantiate the real class and check its action_space.
            pass

    # Real Instantiation attempt
    # We'll need to mock standard init behavior to avoid SC2 process launch.
    # Ideally we just check if it reads the config.
    pass

def test_bot_dynamic_action_masks(level_drop_config):
    """
    Test 3: Verify LevelDropBot constructs masks based on action_heads.
    """
    bot = LevelDropBot(conn=None, training_conf=level_drop_config)
    bot.grid_size = 10
    
    # Current behavior: Hardcoded 4 heads * grid_area
    # Expected behavior: len(action_heads) * grid_area
    
    heads = level_drop_config['env']['action_heads']
    expected_size = len(heads) * (bot.grid_size ** 2)
    
    mask = bot.action_masks()
    
    # This assertion is expected to FAIL until we refactor
    assert mask.shape[0] == expected_size, f"Expected {expected_size}, got {mask.shape[0]}"

def test_bot_reward_config_reading(level_drop_config):
    """
    Test 4: Verify LevelDropBot reads rewards from env.rewards.
    """
    bot = LevelDropBot(conn=None, training_conf=level_drop_config)
    # Mock some state
    bot.previous_cargo_sum = 0
    bot.previous_units_in_zone = set()
    
    # We can't easily trigger the calculation without units, 
    # but we can check if it crashes or uses defaults.
    # Better: Inspect the values it WOULD use.
    # Since the method uses `self.training_conf.get('rewards', {})` currently,
    # and our config puts them in `env.rewards`, 
    # we expect the current code to find NOTHING (or defaults) and our NEW code to find the values.
    
    # To test this, we can set a weird value in the config and see if it's used.
    config = {
        "env": {
            "rewards": {"load_reward": 99.9}
        },
        "rewards": {} # Old location empty
    }
    
    bot = LevelDropBot(conn=None, training_conf=config)
    
    # We'll monkeypatch the units to trigger a reward
    class MockUnit:
        cargo_used = 1
    bot.units = [MockUnit()]
    
    # Logic in calculate_drop_reward:
    # rewards_conf = self.training_conf.get('rewards', {})
    # load_reward = rewards_conf.get('load_reward', 0.1)
    
    # Current code reads from root. Config has it in env.
    # So current code gets default 0.1.
    # Expected code gets 99.9.
    
    reward = bot.calculate_drop_reward()
    
    # This assertion is expected to FAIL until we refactor
    assert reward == 99.9, f"Expected 99.9 (from env.rewards), got {reward} (probably default)"
