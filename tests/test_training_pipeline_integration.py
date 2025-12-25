import pytest
import os
import sys
import shutil
import yaml
import numpy as np
import asyncio
from unittest.mock import MagicMock, AsyncMock
from stable_baselines3.common.monitor import Monitor
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.envs.level_0_env import Level0Env
from train.train_pipeline import PipelineManager

class MockSize:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __getitem__(self, index):
        return (self.x, self.y)[index]

    @property
    def width(self): return self.x
    @property
    def height(self): return self.y

class MockProto:
    def __init__(self, x=128, y=128, in_bits=False):
        self.size = MockSize(x, y)
        if in_bits:
            # 1 bit per pixel -> x*y bits -> x*y/8 bytes
            self.data = bytes(int(x * y // 8))
        else:
            self.data = bytes(int(x * y))

class MockSC2Process:
    def __init__(self, **kwargs):
        self.ws = MagicMock()
        self._ws = self.ws
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def close(self): pass
    def _clean(self): pass
    async def create_game(self, *args, **kwargs): pass
    async def join_game(self, *args, **kwargs): pass

class MockClient:
    def __init__(self, ws): pass
    async def ping(self): pass
    async def actions(self, actions_list): pass
    async def join_game(self, *args, **kwargs): pass
    async def observation(self, *args, **kwargs): return await self.get_observation()
    async def create_game(self, *args, **kwargs): pass
    async def debug_kill_unit(self, *args, **kwargs): pass
    async def debug_create_unit(self, *args, **kwargs): pass
    async def _send_debug(self): pass
    
    async def get_game_info(self):
        class ResponseGameInfo:
            def __init__(self):
                self.game_info = MagicMock()
                # BotAI uses proto_game_info.game_info.start_raw.pathing_grid
                self.game_info.start_raw.pathing_grid = MockProto(128, 128, in_bits=True)
                
        class GameInfoMock:
            def __init__(self):
                self.map_size = MockSize(128, 128)
                self.player_races = {1: Race.Zerg, 2: Race.Zerg}
                self._proto = ResponseGameInfo()
        return GameInfoMock()

    async def get_game_data(self): return None
        
    async def get_observation(self):
        class ResponseObservation:
            def __init__(self):
                self.player_result = []
                self.observation = MagicMock()
                self.observation.game_loop = 0
                self.observation.score = MagicMock()
                self.observation.abilities = []
                self.observation.player_common.player_id = 1
                self.observation.player_common.minerals = 1000
                self.observation.player_common.vespene = 1000
                self.observation.player_common.food_used = 10
                self.observation.player_common.food_cap = 200
                
                # Raw data
                raw_data = MagicMock()
                raw_data.map_state.visibility = MockProto(128, 128, in_bits=False)
                raw_data.map_state.creep = MockProto(128, 128, in_bits=True)
                raw_data.player.power_sources = []
                raw_data.player.upgrade_ids = []
                raw_data.effects = []
                
                # Units
                u = MagicMock()
                u.type_id = UnitTypeId.ZERGLING
                u.alliance = 1
                u.tag = 1
                u.position.x = 10.0
                u.position.y = 10.0
                raw_data.units = [u]
                
                self.observation.raw_data = raw_data
                self.observation_raw = raw_data
        
        return ResponseObservation()

    async def step(self): pass
    async def leave(self): pass

class MonitorWithActionMask(Monitor):
    def action_masks(self):
        return self.env.action_masks()
    
    def get_action_masks(self):
        return self.env.action_masks()

@pytest.fixture
def mock_env_fn():
    def _create_env(worker_index=0):
        # Inject our mock classes
        return Level0Env(
            visualize=False, 
            worker_index=worker_index, 
            sc2_process_cls=MockSC2Process, 
            client_cls=MockClient
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

    # We patch Monitor to ensure we can get action masks if needed, though Level0Env handles it.
    monkeypatch.setattr('train.train_pipeline.Monitor', MonitorWithActionMask)
    
    # Inject our mocks into the manager
    manager = PipelineManager(
        config_dir=temp_config_dir, 
        model_dir_base=temp_model_dir, 
        log_dir_base=temp_log_dir,
        sc2_process_cls=MockSC2Process,
        client_cls=MockClient
    )
    
    with open(os.path.join(temp_config_dir, "level_0.yaml"), 'r') as f:
        level_config = yaml.safe_load(f)
    level_config["training"]["training_steps"] = 2
    level_config["training"]["num_envs"] = 1 # Force single env
    level_config["env"]["headless"] = False # Force SyncVecEnv (since headless=False)
    with open(os.path.join(temp_config_dir, "level_0.yaml"), 'w') as f:
        yaml.safe_dump(level_config, f)

    manager.run()

    final_model_path = os.path.join(temp_model_dir, "level_0", "final_model.zip")
    assert os.path.exists(final_model_path)
