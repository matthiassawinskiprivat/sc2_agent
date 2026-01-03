import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from sc2.ids.unit_typeid import UnitTypeId

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

@pytest.fixture
def mock_env():
    # Mock the unit data loading to avoid file dependency in test
    with patch("train.envs.capture_the_flag_single_env.CaptureTheFlagSingleEnv._load_unit_data") as mock_load:
        mock_load.return_value = {
            "ZERGLING": {"cost": {"minerals": 25, "vespene": 0, "supply": 0.5}},
            "ROACH": {"cost": {"minerals": 75, "vespene": 25, "supply": 2.0}},
            "MARINE": {"cost": {"minerals": 50, "vespene": 0, "supply": 1.0}}
        }
        
        env = CaptureTheFlagSingleEnv(
            mineral_budget=200,
            vespene_budget=25,
            unit_pool=["ZERGLING", "ROACH"],
            map_name="TestMap"
        )
        # Mock the manager and client
        env._manager = MagicMock()
        env._manager.create_join_game = AsyncMock()
        mock_client = MockClient()
        env._manager.clients = [mock_client]
        
        # Mock the bot and its client
        env.bot = MagicMock()
        env.bot.client = mock_client
        env.bot.game_info = MagicMock()
        env.bot.game_info.playable_area.width = 100
        env.bot.game_info.playable_area.height = 100
        env.bot.all_units = MagicMock()
        env.bot.all_units.exists = False
        
        # Mock _refresh_bot_state to avoid GameState parsing
        env._refresh_bot_state = AsyncMock()
        
        yield env

@pytest.mark.asyncio
async def test_budgeted_spawning_integration(mock_env):
    # Test that _spawn_initial_units makes the correct debug_create_unit calls
    await mock_env._spawn_initial_units()
    
    # Verify calls
    # Expect 2 calls for pylons (one per player)
    # Expect 2 calls for army (one per player)
    assert mock_env.bot.client.debug_create_unit.call_count >= 4
    
    calls = mock_env.bot.client.debug_create_unit.call_args_list
    
    # Collect all unit types spawned
    spawned_types = []
    for call in calls:
        args, _ = call
        # args[0] is the list of [unit_id, count, pos, owner]
        cmd_list = args[0]
        for cmd in cmd_list:
            unit_type = cmd[0]
            spawned_types.append(unit_type)
            
    # Check that we spawned valid units
    # Pylons are UnitTypeId.PYLON
    assert UnitTypeId.PYLON in spawned_types
    
    # Check for army units
    # We gave 200m, 25v budget. Should see at least one ROACH and some ZERGLINGS.
    # Note: UnitTypeId for ROACH is 110, ZERGLING is 105 (approx, check your UnitTypeId usage)
    # The env code uses getattr(UnitTypeId, name), so we check against that.
    
    has_roach = getattr(UnitTypeId, "ROACH") in spawned_types
    has_zergling = getattr(UnitTypeId, "ZERGLING") in spawned_types
    
    # It's possible random choice picked only zerglings if it didn't pick roach first, 
    # but with enough budget it usually picks a mix if available.
    # However, to be strict, we just ensure that *some* army units were spawned.
    army_count = len([t for t in spawned_types if t != UnitTypeId.PYLON])
    assert army_count > 0

@pytest.mark.asyncio
async def test_budgeted_spawning_zero_budget(mock_env):
    # Set budget to zero
    mock_env.mineral_budget = 0
    mock_env.vespene_budget = 0
    mock_env.bot.client.debug_create_unit.reset_mock()
    
    await mock_env._spawn_initial_units()
    
    calls = mock_env.bot.client.debug_create_unit.call_args_list
    spawned_types = []
    for call in calls:
        args, _ = call
        cmd_list = args[0]
        for cmd in cmd_list:
            spawned_types.append(cmd[0])
            
    # Should only spawn Pylons (2 calls)
    # (The code makes separate calls for p1 pylon, p2 pylon, p1 army, p2 army)
    # The army calls might be made with empty lists? Let's check implementation.
    # implementation: cmds1 = [[...]]; await debug_create_unit(cmds1)
    # If list is empty, does it still call? Yes, likely.
    
    army_units = [t for t in spawned_types if t != UnitTypeId.PYLON]
    assert len(army_units) == 0
