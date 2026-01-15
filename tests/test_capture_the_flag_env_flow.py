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
    # Mock Clients with _execute
    client1 = MagicMock()
    client1._execute = AsyncMock()
    client1.get_game_data = AsyncMock()
    client1.step = AsyncMock()
    client1.observation = AsyncMock()
    client1.debug_show_map = AsyncMock()
    
    client2 = MagicMock()
    client2._execute = AsyncMock()
    client2.get_game_data = AsyncMock()
    client2.step = AsyncMock()
    client2.observation = AsyncMock()
    client2.debug_show_map = AsyncMock()
    
    mock_manager.clients = [client1, client2]
    env._manager = mock_manager
    
    # Mock bot spawning
    with patch('train.envs.capture_the_flag_env.CaptureTheFlagBot', side_effect=[
        MockCTFBot(player_id=1), MockCTFBot(player_id=2)
    ]) as mock_bot_cls:
        
        # Mock maps.get
        from sc2.data import Race
        mock_game_info = MagicMock()
        mock_game_info.player_races = {1: Race.Zerg, 2: Race.Zerg}
        
        with patch('sc2.maps.get', return_value=MagicMock()), \
             patch('sc2.game_info.GameInfo', return_value=mock_game_info):
            # Mock _synchronized_step and _refresh_bot_states to avoid PixelMap errors
            with patch.object(env, '_synchronized_step', new_callable=AsyncMock), \
                 patch.object(env, '_refresh_bot_states', new_callable=AsyncMock), \
                 patch.object(env, '_spawn_initial_units', new_callable=AsyncMock):
                env.reset()
                assert env.bot1 is not None
                assert env.bot2 is not None

@pytest.mark.asyncio
async def test_ctf_env_step_flow():
    """Test the step flow and reward calculation."""
    env = CaptureTheFlagEnv()
    env.bot1 = MockCTFBot(player_id=1)
    env.bot2 = MockCTFBot(player_id=2)
    
    # Mock bot methods
    env.bot1.client = AsyncMock()
    env.bot2.client = AsyncMock()
    
    # Mock pylon structures
    pylon_list = MagicMock()
    pylon_list.amount = 1
    env.bot1.structures.return_value = pylon_list
    env.bot2.structures.return_value = pylon_list
    
    # Mock unit lists
    u1 = MagicMock(); u1.tag = 1; u1.is_structure = False
    u2 = MagicMock(); u2.tag = 2; u2.is_structure = False
    
    env.bot1.units = MagicMock()
    env.bot1.units.__iter__.return_value = [u1, u1]
    env.bot1.units.filter.return_value = MagicMock(amount=2)
    
    env.bot2.units = MagicMock()
    env.bot2.units.__iter__.return_value = [u2]
    env.bot2.units.filter.return_value = MagicMock(amount=1)
    
    env.active_player_id = 1
    env.current_unit_index = 0
    await env._refresh_unit_lists()
    
    # Mock reward
    env.bot1.calculate_reward.return_value = 1.5
    
    # Mock methods to avoid PixelMap errors
    with patch.object(env, '_refresh_bot_states', new_callable=AsyncMock), \
         patch.object(env, '_synchronized_step', new_callable=AsyncMock), \
         patch.object(env, '_check_termination', return_value=(False, False, 0.0)):
        
        # Test _async_step directly
        obs, reward, done, truncated, info = await env._async_step(10)
        
        assert reward == 1.5
        assert env.active_player_id == 1 
        assert env.current_unit_index == 1
        
        # Step again to switch player
        obs, reward, done, truncated, info = await env._async_step(20)
        assert env.active_player_id == 2
        assert env.current_unit_index == 0
