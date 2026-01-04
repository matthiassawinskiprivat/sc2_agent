import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockClient:
    def __init__(self):
        self.debug_create_unit = AsyncMock()
        self.debug_kill_unit = AsyncMock()
        self.debug_set_unit_value = AsyncMock()
        self.debug_show_map = AsyncMock()
        self.debug_control_enemy = AsyncMock()
        self.step = AsyncMock()
        self.actions = AsyncMock()
        self.observation = AsyncMock()
        self.get_game_data = AsyncMock()

    async def _execute(self, **kwargs):
        mock_resp = MagicMock()
        mock_resp.game_info = MagicMock()
        return mock_resp

def setup_mock_env(env):
    env._manager = MagicMock()
    env._manager.create_join_game = AsyncMock()
    mock_client = MockClient()
    env._manager.clients = [mock_client, MockClient()]
    
    if hasattr(env, 'bot1'):
        env.bot1 = MagicMock()
        env.bot1.client = env._manager.clients[0]
        env.bot1.game_info = MagicMock()
        env.bot1.game_info.playable_area.width = 100
        env.bot1.game_info.playable_area.height = 100
        env.bot1.all_units = MagicMock()
        env.bot1.all_units.exists = False
        env.bot1.structures = MagicMock()
        env.bot1.enemy_structures = MagicMock()
    else:
        env.bot = MagicMock()
        env.bot.client = env._manager.clients[0]
        env.bot.game_info = MagicMock()
        env.bot.game_info.playable_area.width = 100
        env.bot.game_info.playable_area.height = 100
        env.bot.all_units = MagicMock()
        env.bot.all_units.exists = False
        env.bot.structures = MagicMock()
        env.bot.enemy_structures = MagicMock()

    env._refresh_bot_states = AsyncMock()
    env._refresh_bot_state = AsyncMock()
    env._synchronized_step = AsyncMock()
    env._refresh_unit_lists = AsyncMock()

@pytest.mark.asyncio
async def test_multi_env_randomization():
    env = CaptureTheFlagEnv(mineral_budget=500, vespene_budget=0)
    setup_mock_env(env)
    
    positions = []
    for _ in range(10):
        env.bot1.client.debug_create_unit.reset_mock()
        await env._spawn_initial_units()
        
        # Capture Pylon positions from ALL debug_create_unit calls
        p1_pos = None
        p2_pos = None
        for call in env.bot1.client.debug_create_unit.call_args_list:
            call_args = call[0][0]
            for cmd in call_args:
                if cmd[0] == UnitTypeId.PYLON:
                    if cmd[3] == 1: p1_pos = cmd[2]
                    elif cmd[3] == 2: p2_pos = cmd[2]
        
        assert p1_pos is not None and p2_pos is not None, "Both pylons must be spawned"
        positions.append((p1_pos, p2_pos))
        assert p1_pos.distance_to(p2_pos) >= env.flag_distance_min - 1.0
        
    # Verify that not all positions are the same
    assert len(set((p[0].x, p[0].y) for p in positions)) > 1
    assert len(set((p[1].x, p[1].y) for p in positions)) > 1

@pytest.mark.asyncio
async def test_single_env_randomization():
    env = CaptureTheFlagSingleEnv(mineral_budget=500, vespene_budget=0)
    setup_mock_env(env)
    
    positions = []
    for _ in range(10):
        env.bot.client.debug_create_unit.reset_mock()
        await env._spawn_initial_units()
        
        p1_pos = None
        p2_pos = None
        for call in env.bot.client.debug_create_unit.call_args_list:
            call_args = call[0][0]
            for cmd in call_args:
                if cmd[0] == UnitTypeId.PYLON:
                    if cmd[3] == 1: p1_pos = cmd[2]
                    elif cmd[3] == 2: p2_pos = cmd[2]
        
        assert p1_pos is not None and p2_pos is not None, "Both pylons must be spawned"
        positions.append((p1_pos, p2_pos))
        assert p1_pos.distance_to(p2_pos) >= 20.0 - 1.0 # Default in SingleEnv is 20.0
        
    assert len(set((p[0].x, p[0].y) for p in positions)) > 1
    assert len(set((p[1].x, p[1].y) for p in positions)) > 1
