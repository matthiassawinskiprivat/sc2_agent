import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId
from src.agent.ctf_bot import CaptureTheFlagBot

class MockCTFBot(CaptureTheFlagBot):
    def __init__(self, *args, **kwargs):
        super().__init__(player_id=kwargs.get('player_id', 1))
        self.on_start = AsyncMock()
        self.apply_action = AsyncMock()
        self.calculate_reward = MagicMock(return_value=0.0)
        self.get_observation = MagicMock(return_value=np.zeros((7, 128, 128)))
        self.client = AsyncMock()
        self.actions = []
        self._actions = []
        # Mock units properly
        self.units = MagicMock()
        self.structures = MagicMock()
        self.enemy_structures = MagicMock()
        self.game_info = MagicMock()
        self.game_info.playable_area.width = 100
        self.game_info.playable_area.height = 100

def test_ctf_env_full_reset_flow():
    """Test the full reset flow with mocked manager."""
    env = CaptureTheFlagEnv(map_name="TestMap")
    
    # Mock Manager
    mock_manager = MagicMock()
    mock_manager.launch_processes = AsyncMock()
    mock_manager.create_join_game = AsyncMock()
    mock_manager.close = AsyncMock()
    mock_manager.clients = [MagicMock(), MagicMock()]
    env._manager = mock_manager
    
    # Mock bot spawning
    with patch('train.envs.capture_the_flag_env.CaptureTheFlagBot', side_effect=[
        MockCTFBot(player_id=1), MockCTFBot(player_id=2)
    ]) as mock_bot_cls:
        
        # Mock maps.get
        with patch('sc2.maps.get', return_value=MagicMock()):
            # Ensure units exist for initial unit list refresh
            env.reset()
            assert env.bot1 is not None
            assert env.bot2 is not None

@pytest.mark.asyncio
async def test_ctf_env_step_flow():
    """Test the step flow and reward calculation."""
    env = CaptureTheFlagEnv()
    env.bot1 = MockCTFBot(player_id=1)
    env.bot2 = MockCTFBot(player_id=2)
    
    # Mock pylon structures
    pylon_list = MagicMock()
    pylon_list.amount = 1
    env.bot1.structures.return_value = pylon_list
    env.bot2.structures.return_value = pylon_list
    
    # Mock unit lists
    u1 = MagicMock(); u1.tag = 1; u1.is_structure = False
    u2 = MagicMock(); u2.tag = 2; u2.is_structure = False
    
    env.bot1.units = [u1, u1] # Two units for P1
    env.bot2.units = [u2]     # One unit for P2
    
    env.active_player_id = 1
    env.current_unit_index = 0
    await env._refresh_unit_lists()
    
    # Mock reward
    env.bot1.calculate_reward.return_value = 1.5
    
    # Test _async_step directly
    obs, reward, done, truncated, info = await env._async_step(10)
    
    assert reward == 1.5
    assert env.active_player_id == 1 
    assert env.current_unit_index == 1
    
    # Step again to switch player
    obs, reward, done, truncated, info = await env._async_step(20)
    assert env.active_player_id == 2
    assert env.current_unit_index == 0