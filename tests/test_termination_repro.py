import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from train.envs.level_3_env import Level3Env
from sc2.ids.unit_typeid import UnitTypeId

@pytest.fixture
def mock_training_conf():
    return {
        "targets": 2,
        "grid_size": 32,
        "max_steps": 100,
        "rewards": {
            "success": 100.0,
            "distance_multiplier": 1.0,
            "step_penalty": -0.1
        },
        "active_layers": []
    }

class MockUnit:
    def __init__(self, tag):
        self.tag = tag

@pytest.mark.asyncio
@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
async def test_level_3_premature_termination_cycle_end(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Verify that Level3Env DOES NOT return done=True just because a cycle ends,
    if units and beacons still exist.
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": {"targets": 2, "rewards": mock_training_conf["rewards"]}}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    env = Level3Env(level_name="level_3", training_conf=mock_training_conf)
    mock_bot = MagicMock()
    mock_bot.apply_action = AsyncMock()
    env._bot = mock_bot
    
    # 2 units exist
    unit1 = MockUnit(101)
    unit2 = MockUnit(102)
    mock_bot.get_active_zerglings.return_value = [unit1, unit2]
    
    # Beacons still exist
    mock_beacons = MagicMock()
    mock_beacons.amount = 2
    mock_bot.units.return_value = mock_beacons
    
    mock_bot.get_observation_for_unit.return_value = (np.zeros((1, 32, 32)), dict())
    mock_bot.calculate_distance_reward.return_value = 1.0
    mock_bot.check_collections = AsyncMock(return_value=dict()) # No collections
    
    env._client = MagicMock()
    env._client.step = AsyncMock()
    env._client.observation = AsyncMock()
    
    with patch("sc2.game_state.GameState"), \
         patch("train.envs.level_0_env.ProtoWrapper"):
        env._bot._prepare_step = MagicMock()
        env.active_unit_tags = [101, 102]
        env.current_unit_index = 0
        
        # Step 1: Unit 101 acts
        obs1, reward1, done1, truncated1, info1 = await env._async_step(action=1)
        assert not done1, "Should not be done after first unit"
        assert env.current_unit_index == 1
        
        # Step 2: Unit 102 acts (Cycle ends)
        obs2, reward2, done2, truncated2, info2 = await env._async_step(action=1)
        
        # This is where it likely fails currently
        assert not done2, "Should NOT be done just because cycle ended if units/beacons remain"
        assert env.current_unit_index == 0, "Cycle should reset to 0"

@pytest.mark.asyncio
@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
async def test_level_3_premature_termination_unit_death(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Verify that Level3Env DOES NOT return done=True if a unit dies mid-cycle
    but other units remain.
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": {"targets": 2, "rewards": mock_training_conf["rewards"]}}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    env = Level3Env(level_name="level_3", training_conf=mock_training_conf)
    mock_bot = MagicMock()
    mock_bot.apply_action = AsyncMock()
    env._bot = mock_bot
    
    # Unit 101 dies after Step 1
    unit1 = MockUnit(101)
    unit2 = MockUnit(102)
    
    # 1st call (init active tags): [101, 102]
    # 2nd call (after unit 101 step): [102]
    # 3rd call (termination check): [102]
    mock_bot.get_active_zerglings.side_effect = [
        [unit1, unit2], 
        [unit2],
        [unit2]
    ]
    
    # Beacons still exist
    mock_beacons = MagicMock()
    mock_beacons.amount = 2
    mock_bot.units.return_value = mock_beacons
    
    mock_bot.get_observation_for_unit.return_value = (np.zeros((1, 32, 32)), dict())
    mock_bot.calculate_distance_reward.return_value = 1.0
    mock_bot.check_collections = AsyncMock(return_value=dict())
    
    env._client = MagicMock()
    env._client.step = AsyncMock()
    env._client.observation = AsyncMock()
    
    with patch("sc2.game_state.GameState"), \
         patch("train.envs.level_0_env.ProtoWrapper"):
        env._bot._prepare_step = MagicMock()
        env.active_unit_tags = [101, 102]
        env.current_unit_index = 0
        
        # Step 1: Unit 101 acts and dies
        obs1, reward1, done1, truncated1, info1 = await env._async_step(action=1)
        
        assert not done1, "Should NOT be done if unit died but others remain"
        # Next unit in original list was 102. 
        # Env should detect 102 is still alive and return its observation.
        mock_bot.get_observation_for_unit.assert_called_with(102)

@pytest.mark.asyncio
@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
async def test_level_3_premature_termination_no_units(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Verify that Level3Env returns done=True if NO units are found.
    (This is expected behavior, but helps confirm if this is what's happening).
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": {"targets": 2, "rewards": mock_training_conf["rewards"]}}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    env = Level3Env(level_name="level_3", training_conf=mock_training_conf)
    mock_bot = MagicMock()
    mock_bot.apply_action = AsyncMock()
    env._bot = mock_bot
    
    # NO units found
    mock_bot.get_active_zerglings.return_value = []
    
    # Beacons still exist
    mock_beacons = MagicMock()
    mock_beacons.amount = 2
    mock_bot.units.return_value = mock_beacons
    
    env._client = MagicMock()
    env._client.step = AsyncMock()
    env._client.observation = AsyncMock()
    
    with patch("sc2.game_state.GameState"), \
         patch("train.envs.level_0_env.ProtoWrapper"):
        env._bot._prepare_step = MagicMock()
        
        # Scenario: Start with NO units (e.g. wrong allied_unit_type)
        env.active_unit_tags = []
        env.current_unit_index = 0
        
        obs, reward, done, truncated, info = await env._async_step(action=1)
        assert done, "Should be done if no units left"

