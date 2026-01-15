import pytest
import yaml
import os
import numpy as np
from unittest.mock import MagicMock, patch
from src.utils.config_manager import ConfigManager
from train.envs.level_0_env import Level0Env

def test_cumulative_active_layers(tmp_path):
    # Setup mock config files
    pipeline_conf = {'levels': ['lvl0', 'lvl1', 'lvl2']}
    
    lvl0_conf = {'env': {'active_layers': ['A', 'B']}}
    lvl1_conf = {'env': {'active_layers': ['C']}} # Should become [A, B, C]
    lvl2_conf = {'env': {}} # Should remain [A, B, C]
    
    (tmp_path / "pipeline.yaml").write_text(yaml.dump(pipeline_conf))
    (tmp_path / "lvl0.yaml").write_text(yaml.dump(lvl0_conf))
    (tmp_path / "lvl1.yaml").write_text(yaml.dump(lvl1_conf))
    (tmp_path / "lvl2.yaml").write_text(yaml.dump(lvl2_conf))
    
    manager = ConfigManager(str(tmp_path))
    
    assert manager.get_cumulative_active_layers('lvl0') == ['A', 'B']
    assert manager.get_cumulative_active_layers('lvl1') == ['A', 'B', 'C']
    assert manager.get_cumulative_active_layers('lvl2') == ['A', 'B', 'C']

def test_cumulative_active_layers_duplicates(tmp_path):
    # Setup mock config files
    pipeline_conf = {'levels': ['lvl0', 'lvl1']}
    
    lvl0_conf = {'env': {'active_layers': ['A', 'B']}}
    lvl1_conf = {'env': {'active_layers': ['B', 'C']}} # B is duplicate, should be [A, B, C]
    
    (tmp_path / "pipeline.yaml").write_text(yaml.dump(pipeline_conf))
    (tmp_path / "lvl0.yaml").write_text(yaml.dump(lvl0_conf))
    (tmp_path / "lvl1.yaml").write_text(yaml.dump(lvl1_conf))
    
    manager = ConfigManager(str(tmp_path))
    
    assert manager.get_cumulative_active_layers('lvl1') == ['A', 'B', 'C']

def test_cumulative_action_heads(tmp_path):
    # Setup mock config files
    pipeline_conf = {'levels': ['lvl0', 'lvl1', 'lvl2']}
    
    lvl0_conf = {'env': {'action_heads': ['move']}}
    lvl1_conf = {'env': {'action_heads': ['attack']}} # Should become [move, attack]
    lvl2_conf = {'env': {'action_heads': ['patrol']}} # Should become [move, attack, patrol]
    
    (tmp_path / "pipeline.yaml").write_text(yaml.dump(pipeline_conf))
    (tmp_path / "lvl0.yaml").write_text(yaml.dump(lvl0_conf))
    (tmp_path / "lvl1.yaml").write_text(yaml.dump(lvl1_conf))
    (tmp_path / "lvl2.yaml").write_text(yaml.dump(lvl2_conf))
    
    manager = ConfigManager(str(tmp_path))
    
    assert manager.get_cumulative_action_heads('lvl0') == ['move']
    assert manager.get_cumulative_action_heads('lvl1') == ['move', 'attack']
    assert manager.get_cumulative_action_heads('lvl2') == ['move', 'attack', 'patrol']

def test_cumulative_action_heads_off_pipeline(tmp_path):
    # Setup mock config files
    pipeline_conf = {'levels': ['lvl0', 'lvl1']}
    
    lvl0_conf = {'env': {'action_heads': ['move']}}
    lvl1_conf = {'env': {'action_heads': ['attack']}}
    lvl_drop_conf = {'env': {'action_heads': ['load', 'unload']}} # Off-pipeline, should inherit all + self
    
    (tmp_path / "pipeline.yaml").write_text(yaml.dump(pipeline_conf))
    (tmp_path / "lvl0.yaml").write_text(yaml.dump(lvl0_conf))
    (tmp_path / "lvl1.yaml").write_text(yaml.dump(lvl1_conf))
    (tmp_path / "lvl_drop.yaml").write_text(yaml.dump(lvl_drop_conf))
    
    manager = ConfigManager(str(tmp_path))
    
    # Should contain everything from pipeline (move, attack) plus its own (load, unload)
    expected = ['move', 'attack', 'load', 'unload']
    assert manager.get_cumulative_action_heads('lvl_drop') == expected

@patch('train.envs.level_0_env.ConfigManager')
@patch('train.envs.level_0_env.SC2Process')
@patch('train.envs.level_0_env.Client')
def test_level0_env_dynamic_observation_space(mock_client, mock_sc2, mock_config_manager_cls):
    # Mock ConfigManager instance
    mock_instance = MagicMock()
    mock_instance.get_cumulative_active_layers.return_value = ['A', 'B', 'C']
    mock_instance.get_cumulative_action_heads.return_value = ['move']
    mock_config_manager_cls.return_value = mock_instance
    
    # Initialize env
    # Note: We need to provide map_list logic or mock it, otherwise it fails at _discover_maps
    with patch.object(Level0Env, '_discover_maps', return_value=['map1']):
        env = Level0Env(level_name="test_level", visualize=False)
    
    # Check if observation space shape matches 3 layers
    assert env.observation_space.shape == (3, 128, 128)
    assert env.training_conf['active_layers'] == ['A', 'B', 'C']

@patch('train.envs.level_0_env.ConfigManager')
@patch('train.envs.level_0_env.SC2Process')
@patch('train.envs.level_0_env.Client')
def test_level0_env_dynamic_action_space(mock_client, mock_sc2, mock_config_manager_cls):
    # Mock ConfigManager instance
    mock_instance = MagicMock()
    mock_instance.get_cumulative_active_layers.return_value = ['layer1']
    # Simulate 2 action heads
    mock_instance.get_cumulative_action_heads.return_value = ['move', 'extra']
    mock_config_manager_cls.return_value = mock_instance
    
    # Initialize env
    with patch.object(Level0Env, '_discover_maps', return_value=['map1']):
        # Grid size 32 -> grid_area 1024
        env = Level0Env(level_name="test_level", visualize=False, grid_size=32)
    
    # Action space should be heads * area = 2 * 1024 = 2048
    assert env.action_space.n == 2048
    assert env.action_heads == ['move', 'extra']
