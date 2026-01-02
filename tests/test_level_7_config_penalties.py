import yaml
import os

def test_level_7_config_has_penalties():
    config_path = "config/level_7.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    rewards = config.get("env", {}).get("rewards", {})
    
    assert "damage_penalty" in rewards, "damage_penalty missing from config"
    assert "kill_penalty" in rewards, "kill_penalty missing from config"
    assert rewards["damage_penalty"] == -0.5
    assert rewards["kill_penalty"] == -10.0
