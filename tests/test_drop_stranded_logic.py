
import pytest
import numpy as np
import unittest.mock
from unittest.mock import MagicMock, AsyncMock
from train.envs.level_drop_env import LevelDropEnv
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

@pytest.mark.asyncio
async def test_stranded_termination_logic():
    """
    Verifies that the LevelDropEnv terminates when transports are dead
    and ground units have no path to the objective.
    """
    # 1. Setup Environment Configuration
    training_conf = {
        "env": {
            "level_name": "level_drop_0",
            "rewards": {"loss_penalty": -10.0}
        }
    }
    
    # Mock LevelDropEnv to prevent real initialization side effects
    with MagicMock() as mock_cm:
        # Prevent discover_maps from failing
        LevelDropEnv._discover_maps = MagicMock(return_value=["test_map"])
        
        # Initialize env with training_conf
        env = LevelDropEnv(training_conf=training_conf, level_name="level_drop_0")
        
        # Mock dependencies
        env._bot = MagicMock()
        env._bot.apply_action = AsyncMock() # Must be AsyncMock for await
        env._client = AsyncMock()
        
        # Fix: Ensure reward calculations return floats, not MagicMocks
        env._bot.calculate_combat_reward.return_value = 0.0
        env._bot.calculate_distance_reward.return_value = 0.0
        env._bot.calculate_allied_attrition_penalty.return_value = 0.0
        env._bot.calculate_drop_reward.return_value = 0.0

        # Prevent PixelMap assertion error by mocking the GameState creation
        with unittest.mock.patch('sc2.game_state.GameState', MagicMock()):
            # Mock observation results
            dummy_obs = np.zeros((1, 128, 128))
            env._bot.get_observation_for_unit = MagicMock(return_value=(dummy_obs, {}))

            # Setup Unit Type IDs (simulating LevelDropBot constants)
            env._bot.TRANSPORT_TYPES = {UnitTypeId.MEDIVAC}
            env._bot.INFANTRY_TYPES = {UnitTypeId.MARINE}
            env._bot._objective_pos = Point2((100, 100))
            env._bot.enemy_start_locations = []
            env.steps = 0

            # 2. Case A: Transports Alive -> Should NOT terminate
            mock_transport = MagicMock(spec=['type_id', 'tag', 'position', 'distance_to'])
            mock_transport.type_id = UnitTypeId.MEDIVAC
            mock_transport.tag = 1
            mock_transport.position = Point2((10, 10))
            mock_transport.distance_to.return_value = 50.0

            env._bot.get_active_units.return_value = [mock_transport]
            env._bot.enemy_units = MagicMock(amount=1)
            env._bot.enemy_structures = MagicMock(amount=1)
            # Mock townhall filter
            mock_townhalls = MagicMock(amount=1)
            env._bot.enemy_structures.filter.return_value = mock_townhalls

            # Simulate a step
            _, _, done, _, _ = await env._async_step(0)
            assert not done, "Episode should not end while transports are alive"

            # 3. Case B: Transports Dead, Infantry Stranded -> Should TERMINATE
            mock_marine = MagicMock(spec=['type_id', 'tag', 'position', 'distance_to'])
            mock_marine.type_id = UnitTypeId.MARINE
            mock_marine.tag = 2
            mock_marine.position = Point2((10, 10))
            mock_marine.distance_to.return_value = 50.0
            
            env._bot.get_active_units.return_value = [mock_marine]
            
            # Mock query_pathing to return 0.0 (unreachable)
            env._client.query_pathing.return_value = 0.0
            
            # Simulate a step
            # Reset index to match mock infantry
            env.current_unit_index = 0
            env.active_unit_tags = [2]
            _, reward, done, _, _ = await env._async_step(0)
            
            assert done, "Episode should end when all units are stranded"
            assert reward <= -10.0, f"Should receive loss penalty. Reward was: {reward}"
            
            # 4. Case C: Transports Dead, Infantry Reachable -> Should NOT terminate
            env._client.query_pathing.return_value = 25.0 # Path exists
            
            # Reset state for next test
            env.current_unit_index = 0
            env.active_unit_tags = [2]
            
            _, reward, done, _, _ = await env._async_step(0)
            assert not done, "Episode should continue if units can reach the objective"
