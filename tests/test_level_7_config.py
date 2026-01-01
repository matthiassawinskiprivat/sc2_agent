import pytest
import os
import yaml
from src.utils.config_manager import ConfigManager

def test_level_7_config_exists():
    config_path = "config/level_7.yaml"
    assert os.path.exists(config_path), f"Config file {config_path} should exist."

def test_level_7_config_content():
    config_path = "config/level_7.yaml"
    if not os.path.exists(config_path):
        pytest.fail(f"Config file {config_path} does not exist.")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    assert "training" in config
    assert "env" in config
    assert "model" in config
    
    env_conf = config["env"]
    assert env_conf.get("env_class") == "Level7Env"
    assert env_conf.get("unit_type") == "MARINE"
    assert env_conf.get("enemy_type") == "ZERGLING"
    
    rewards = env_conf.get("rewards", {})
    assert "damage_reward" in rewards
    assert "kill_reward" in rewards
