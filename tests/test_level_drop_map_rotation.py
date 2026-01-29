import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from train.envs.level_drop_env import LevelDropEnv

def test_level_drop_map_rotation_logic():
    """
    Verify that LevelDropEnv rotates maps on EVERY reset call.
    """
    # Mock ConfigManager and SC2 dependencies
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = []
    mock_cm.get_cumulative_action_heads.return_value = ["move"]
    mock_cm.get_level_config.return_value = {"env": {}}

    map_list = ['MapA', 'MapB', 'MapC']
    
    with patch('train.envs.level_0_env.ConfigManager', return_value=mock_cm), \
         patch('train.envs.level_0_env.SC2Process'), \
         patch('train.envs.level_0_env.Client'), \
         patch('train.envs.level_0_env.Level0Env._discover_maps', return_value=map_list):
        
        # 1. Initial State (Worker 0)
        env = LevelDropEnv(level_name="level_drop_0", worker_index=0, num_workers=1)
        assert env.map_name == 'MapA'
        
        # Mock super().reset to avoid real SC2 logic
        with patch('train.envs.level_0_env.Level0Env.reset', return_value=(None, {})):
            
            # 2. First Rotation
            env.episode_counter = 1
            env.reset()
            assert env.map_name == 'MapB'
            
            # 3. Second Rotation
            env.episode_counter = 2
            env.reset()
            assert env.map_name == 'MapC'
            
            # 4. Cycle back
            env.episode_counter = 3
            env.reset()
            assert env.map_name == 'MapA'

def test_level_drop_worker_offset_diversity():
    """
    Verify that different worker_index values result in different starting maps.
    """
    map_list = ['MapA', 'MapB', 'MapC']
    
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
        
        assert env0.map_name == 'MapA'
        assert env1.map_name == 'MapB'
        assert env2.map_name == 'MapC'
        assert env3.map_name == 'MapA' # Wraps around
