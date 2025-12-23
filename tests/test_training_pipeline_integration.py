import pytest
import os
import sys
import shutil
import yaml
import numpy as np
import time
from multiprocessing import Process, Pipe
from unittest.mock import MagicMock
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.envs.level_0_env import Level0Env
from train.train_pipeline import PipelineManager

class MockSC2Process:
    def __init__(self, **kwargs):
        self.ws = None
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    def close(self):
        pass

class MockClient:
    def __init__(self, ws):
        pass
    async def create_game(self, *args, **kwargs):
        pass
    async def get_game_info(self):
        class MockGameInfo:
            def __init__(self):
                self.map_size = self
                self.width = 128
                self.height = 128
                self.player_races = {1: Race.Zerg, 2: Race.Zerg}
                
                proto = MagicMock()
                proto.game_info.start_raw.pathing_grid.size.x = 128
                proto.game_info.start_raw.pathing_grid.size.y = 128
                proto.game_info.start_raw.pathing_grid.data = bytes(2048)
                self._proto = proto

        return MockGameInfo()
    async def get_game_data(self):
        return None
    async def get_observation(self):
        class MockObservation:
            def __init__(self):
                self.actions = []
                self.observation = self
                
                mock_zergling = MagicMock()
                mock_zergling.type_id = UnitTypeId.ZERGLING
                mock_zergling.alliance = 1
                mock_zergling.tag = 1
                
                self.observation_raw = MagicMock()
                self.observation_raw.units = [mock_zergling]
                self.raw_data = self.observation_raw

                self.game_loop = 0
                self.common = MagicMock()
        return MockObservation()
    async def step(self):
        pass
    async def leave(self):
        pass

class MonitorWithActionMask(Monitor):
    def get_action_masks(self):
        return self.env.get_action_masks()

@pytest.fixture
def mock_env_fn(monkeypatch):
    """Fixture to create a Level0Env with mocked SC2Process and Client."""
    monkeypatch.setattr("train.envs.level_0_env.SC2Process", MockSC2Process)
    monkeypatch.setattr("train.envs.level_0_env.Client", MockClient)
    def _create_env(worker_index=0):
        return Level0Env(
            visualize=False, 
            worker_index=worker_index
        )
    return _create_env

def test_level0env_with_mocked_sc2(mock_env_fn):
    """Tests the Level0Env with mocked SC2Process and Client."""
    env = mock_env_fn()
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    
    env.close()

@pytest.fixture(scope="module")
def temp_project_setup():
    """Fixture for setting up a temporary project structure for pipeline tests."""
    temp_config_dir = "temp_config"
    temp_model_dir = "temp_models"
    temp_log_dir = "temp_logs"
    original_config_dir = "config"

    os.makedirs(temp_config_dir, exist_ok=True)
    os.makedirs(temp_model_dir, exist_ok=True)
    os.makedirs(temp_log_dir, exist_ok=True)

    # Copy and modify pipeline.yaml
    with open(os.path.join(original_config_dir, "pipeline.yaml"), 'r') as f:
        pipeline_config = yaml.safe_load(f)
    pipeline_config["levels"] = ["level_0"] # Only run level_0 for this test
    with open(os.path.join(temp_config_dir, "pipeline.yaml"), 'w') as f:
        yaml.safe_dump(pipeline_config, f)

    # Copy level_0.yaml
    shutil.copy(os.path.join(original_config_dir, "level_0.yaml"), temp_config_dir)

    yield temp_config_dir, temp_model_dir, temp_log_dir

    shutil.rmtree(temp_config_dir)
    shutil.rmtree(temp_model_dir)
    shutil.rmtree(temp_log_dir)

def test_pipeline_manager_with_mock(temp_project_setup, monkeypatch):
    """Tests that the PipelineManager can run a short training loop with a mocked env."""
    temp_config_dir, temp_model_dir, temp_log_dir = temp_project_setup

    monkeypatch.setattr("train.envs.level_0_env.SC2Process", MockSC2Process)
    monkeypatch.setattr("train.envs.level_0_env.Client", MockClient)
    monkeypatch.setattr('train.train_pipeline.Monitor', MonitorWithActionMask)
    
    manager = PipelineManager(config_dir=temp_config_dir, model_dir_base=temp_model_dir, log_dir_base=temp_log_dir)
    
    with open(os.path.join(temp_config_dir, "level_0.yaml"), 'r') as f:
        level_config = yaml.safe_load(f)
    level_config["training"]["training_steps"] = 20
    level_config["training"]["num_envs"] = 2
    with open(os.path.join(temp_config_dir, "level_0.yaml"), 'w') as f:
        yaml.safe_dump(level_config, f)

    manager.run()

    final_model_path = os.path.join(temp_model_dir, "level_0", "final_model.zip")
    assert os.path.exists(final_model_path)
