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
    
    # Use realistic offsets from BotMicroArena_6 inspection
    # x=2, y=2, w=68, h=52 -> Top Right is (70, 54)
    def setup_bot_mock(bot):
        bot.client = env._manager.clients[0]
        bot.game_info = MagicMock()
        bot.game_info.playable_area.x = 2.0
        bot.game_info.playable_area.y = 2.0
        bot.game_info.playable_area.width = 68.0
        bot.game_info.playable_area.height = 52.0
        bot.game_info.map_size.width = 100
        bot.game_info.map_size.height = 100
        
        # Mock pathing_grid
        bot.game_info.pathing_grid = MagicMock()
        bot.game_info.pathing_grid.width = 100
        bot.game_info.pathing_grid.height = 100
        # __getitem__ returns 1 (pathable) for any coordinate
        bot.game_info.pathing_grid.__getitem__.return_value = 1
        
        bot.all_units = MagicMock()
        bot.all_units.exists = False
        bot.structures = MagicMock()
        bot.enemy_structures = MagicMock()

    if hasattr(env, 'bot1'):
        env.bot1 = MagicMock()
        setup_bot_mock(env.bot1)
        env.bot2 = MagicMock()
        setup_bot_mock(env.bot2)
    else:
        env.bot = MagicMock()
        setup_bot_mock(env.bot)

    env._refresh_bot_states = AsyncMock()
    env._refresh_bot_state = AsyncMock()
    env._synchronized_step = AsyncMock()
    env._refresh_unit_lists = AsyncMock()

@pytest.mark.asyncio
async def test_multi_env_randomization():
    env = CaptureTheFlagEnv(mineral_budget=500, vespene_budget=0)
    setup_mock_env(env)
    
    positions = []
    # Use more iterations to ensure coverage
    for _ in range(50):
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
        
    # Verify coverage across map
    all_p_coords = [p[0] for p in positions] + [p[1] for p in positions]
    max_x = max(p.x for p in all_p_coords)
    max_y = max(p.y for p in all_p_coords)
    min_x = min(p.x for p in all_p_coords)
    min_y = min(p.y for p in all_p_coords)
    
    # Offsets are 2.0, w=68, h=52 -> Full range is [2, 70] and [2, 54]
    # With margin 1.0, range is [3, 69] and [3, 53]
    # We should see points significantly above the old hardcoded limits (relative coords without offset)
    assert max_x > 65.0, f"Spawning didn't reach top area (x={max_x})"
    assert max_y > 50.0, f"Spawning didn't reach right area (y={max_y})"
    assert min_x < 10.0
    assert min_y < 10.0

@pytest.mark.asyncio
async def test_single_env_randomization():
    env = CaptureTheFlagSingleEnv(mineral_budget=500, vespene_budget=0)
    setup_mock_env(env)
    
    positions = []
    for _ in range(50):
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
        
    all_p_coords = [p[0] for p in positions] + [p[1] for p in positions]
    max_x = max(p.x for p in all_p_coords)
    max_y = max(p.y for p in all_p_coords)
    
    assert max_x > 65.0, f"Spawning didn't reach top area (x={max_x})"
    assert max_y > 50.0, f"Spawning didn't reach right area (y={max_y})"
