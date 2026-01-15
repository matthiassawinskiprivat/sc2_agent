import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from train.envs.level_drop_env import LevelDropEnv

def verify_init():
    print("Verifying LevelDropEnv initialization...")
    
    # Mock ConfigManager and SC2 dependencies
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move"]
    mock_cm.get_level_config.return_value = {"env": {}}

    # Mock discover maps to return something
    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.SC2Process'), \
         patch('train.envs.level_0_env.Client'), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=['MapA', 'MapB']):
        
        # Instantiate without map_name
        env = LevelDropEnv(level_name="level_drop_0", worker_index=0, num_workers=1)
        
        print(f"Map list: {env.map_list}")
        print(f"Initial map: {env.map_name}")
        
        assert env.map_name == 'MapA'
        assert env.level_name == 'level_drop_0'
        print("Initialization SUCCESS")

if __name__ == "__main__":
    verify_init()
