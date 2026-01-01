import pytest
from unittest.mock import MagicMock, patch, ANY
import yaml
import os
from train.train_pipeline import PipelineManager
from train.envs.level_0_env import Level0Env
import gymnasium as gym

# Dummy environment class to test dynamic loading
class DummyEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 64, 64))
        self.action_space = gym.spaces.Discrete(10)
    
    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}
    
    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

@patch("train.train_pipeline.MaskablePPO")
@patch("train.train_pipeline.MaskableSubprocVecEnv")
@patch("train.train_pipeline.Monitor")
def test_pipeline_selects_correct_env_class(mock_monitor, mock_vec_env, mock_ppo):
    """
    Test that the pipeline selects the correct environment class based on configuration.
    """
    # Configure mock_vec_env
    mock_vec_env.__name__ = "MaskableSubprocVecEnv"
    
    # Mock configuration
    mock_pipeline_config = {
        "levels": ["level_test"]
    }
    
    mock_level_config = {
        "training": {
            "training_steps": 100,
            "num_envs": 1,
            "n_steps": 128,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "env": {
            "grid_size": 32,
            "headless": True,
            "env_class": "DummyEnv", # Target class
            "max_steps": 100
        },
        "model": {
            "policy": "CnnPolicy",
            "policy_kwargs": {},
            "gamma": 0.99,
            "ent_coef": 0.01,
            "n_epochs": 4
        },
        "evaluation": {}
    }
    
    # Patch ENV_REGISTRY to include DummyEnv
    with patch.dict("train.train_pipeline.ENV_REGISTRY", {"DummyEnv": DummyEnv}):
        # Patch yaml.safe_load to return our mocks
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            # Set name to satisfy side_effect check
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch("yaml.safe_load") as mock_yaml_load:
                def side_effect(f):
                    # We can't rely on f.name easily with mocks sometimes, 
                    # so let's rely on call count or context.
                    # First call is pipeline, second is level.
                    if mock_yaml_load.call_count == 1:
                        return mock_pipeline_config
                    return mock_level_config
                
                mock_yaml_load.side_effect = side_effect
                
                # Initialize pipeline
                with patch("glob.glob", return_value=["config/level_test.yaml"]):
                    manager = PipelineManager(config_dir="config")
                    
                    manager._train_level("level_test")
                    
                    if mock_vec_env.call_args:
                        env_fns = mock_vec_env.call_args[0][0]
                        # Create the env
                        created_env = env_fns[0]()
                        
                        # Inspect the Monitor's env argument (the wrapped env)
                        wrapped_env = mock_monitor.call_args[0][0]
                        
                        # Verify it is an instance of DummyEnv
                        assert isinstance(wrapped_env, DummyEnv), \
                            f"Expected DummyEnv, got {type(wrapped_env)}"
                    else:
                        pytest.fail("SubprocVecEnv was not instantiated")