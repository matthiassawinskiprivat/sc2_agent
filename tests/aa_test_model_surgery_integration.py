import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from train.train_pipeline import PipelineManager

class SimpleEnv(gym.Env):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(n_actions)
    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}
    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, False, False, {}

@pytest.fixture
def mock_l0_model(tmp_path):
    env = SimpleEnv((4, 128, 128), 256)
    model = MaskablePPO("CnnPolicy", env, verbose=0)
    path = tmp_path / "l0_model.zip"
    model.save(str(path))
    return str(path)

@patch('train.train_pipeline.Level0Env')
@patch('train.train_pipeline.Monitor')
@patch('train.train_pipeline.MaskableSubprocVecEnv')
@patch('train.train_pipeline.yaml.safe_load')
def test_train_level_with_obs_mismatch(mock_yaml, mock_vec_env, mock_monitor, mock_level_env, mock_l0_model):
    # 1. Setup L1 config (6 channels)
    mock_yaml.side_effect = [
        {'levels': ['level_1']}, # pipeline.yaml
        { # level_1.yaml
            'training': {'num_envs': 1, 'learning_rate': 0.001, 'batch_size': 64, 'n_steps': 128, 'training_steps': 1, 'num_envs': 1},
            'env': {'headless': True, 'grid_size': 32},
            'model': {'policy': 'CnnPolicy', 'gamma': 0.99, 'ent_coef': 0.01, 'n_epochs': 1}
        }
    ]
    
    # 2. Setup L1 Env (6 channels)
    env_l1 = SimpleEnv((6, 128, 128), 1024)
    # SB3 needs num_envs on vec_envs
    env_l1.num_envs = 1
    
    mock_vec_env.return_value = env_l1
    mock_vec_env.__name__ = "MockVecEnv"
    
    manager = PipelineManager(config_dir="config")
    
    # 3. Execution
    # We want to verify that the surgery works and the model loads without error
    with patch.object(MaskablePPO, 'learn'):
        with patch.object(MaskablePPO, 'save'):
            # This should now succeed without RuntimeError
            manager._train_level("level_1", model_path_to_load=mock_l0_model)
            
            # If we reached here, it means load_state_dict succeeded
