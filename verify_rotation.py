import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from train.envs.level_drop_env import LevelDropEnv

def verify_rotation():
    print("Verifying LevelDropEnv map rotation...")
    
    # Mock ConfigManager and SC2 dependencies
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move"]
    mock_cm.get_level_config.return_value = {"env": {}}

    # Mock discover maps to return multiple maps
    map_list = ['MapA', 'MapB', 'MapC']
    
    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.SC2Process'), \
         patch('train.envs.level_0_env.Client'), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=map_list):
        
        # 1. Initial State (Worker 0)
        env = LevelDropEnv(level_name="level_drop_0", worker_index=0, num_workers=1)
        print(f"Initial map: {env.map_name}")
        assert env.map_name == 'MapA'
        
        # Mock super().reset to avoid real SC2 logic
        with patch('train.envs.level_0_env.Level0Env.reset', return_value=(None, {})):
            
            # 2. First Rotation
            env.episode_counter = 1
            env.reset()
            print(f"After reset 1: {env.map_name}")
            assert env.map_name == 'MapB'
            
            # 3. Second Rotation
            env.reset()
            print(f"After reset 2: {env.map_name}")
            assert env.map_name == 'MapC'
            
            # 4. Cycle back
            env.reset()
            print(f"After reset 3 (cycle): {env.map_name}")
            assert env.map_name == 'MapA'

    print("Rotation SUCCESS")

def verify_worker_offset():
    print("Verifying Worker Offset diversity...")
    map_list = ['MapA', 'MapB', 'MapC']
    
    # Mocking same as above
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move"]
    mock_cm.get_level_config.return_value = {"env": {}}

    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.SC2Process'), \
         patch('train.envs.level_0_env.Client'), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=map_list):
        
        env0 = LevelDropEnv(level_name="level_drop_0", worker_index=0, num_workers=4)
        env1 = LevelDropEnv(level_name="level_drop_0", worker_index=1, num_workers=4)
        env2 = LevelDropEnv(level_name="level_drop_0", worker_index=2, num_workers=4)
        env3 = LevelDropEnv(level_name="level_drop_0", worker_index=3, num_workers=4)
        
        print(f"Worker 0 starts on: {env0.map_name}")
        print(f"Worker 1 starts on: {env1.map_name}")
        print(f"Worker 2 starts on: {env2.map_name}")
        print(f"Worker 3 starts on: {env3.map_name}")
        
        assert env0.map_name == 'MapA'
        assert env1.map_name == 'MapB'
        assert env2.map_name == 'MapC'
        assert env3.map_name == 'MapA' # Wraps around

    print("Worker Offset SUCCESS")

if __name__ == "__main__":
    verify_rotation()
    verify_worker_offset()
