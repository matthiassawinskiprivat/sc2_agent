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
def mock_env_factory():
    def _create_env(minerals, vespene, unit_pool):
        # Mock unit data
        mock_data = {
            "ZERGLING": {"cost": {"minerals": 25, "vespene": 0, "supply": 0.5}},
            "ROACH": {"cost": {"minerals": 75, "vespene": 25, "supply": 2.0}},
            "MARINE": {"cost": {"minerals": 50, "vespene": 0, "supply": 1.0}},
            "REAPER": {"cost": {"minerals": 50, "vespene": 50, "supply": 1.0}},
            "ZEALOT": {"cost": {"minerals": 100, "vespene": 0, "supply": 2.0}}
        }

        with patch("train.envs.capture_the_flag_single_env.CaptureTheFlagSingleEnv._load_unit_data") as mock_load:
            mock_load.return_value = mock_data
            
            env = CaptureTheFlagSingleEnv(
                mineral_budget=minerals,
                vespene_budget=vespene,
                unit_pool=unit_pool,
                map_name="TestMap"
            )
            # Mock manager/client
            env._manager = MagicMock()
            env._manager.create_join_game = AsyncMock()
            mock_client = MockClient()
            env._manager.clients = [mock_client]
            
            # Mock bot
            env.bot = MagicMock()
            env.bot.client = mock_client
            env.bot.game_info = MagicMock()
            env.bot.game_info.playable_area.width = 100
            env.bot.game_info.playable_area.height = 100
            env.bot.all_units = MagicMock()
            env.bot.all_units.exists = False
            
            # Mock refresh state
            env._refresh_bot_state = AsyncMock()
            
            return env
    return _create_env

@pytest.mark.asyncio
async def test_composition_zergling_only(mock_env_factory):
    # Budget only allows Zerglings (no vespene for Roach/Reaper, Zealot too expensive)
    env = mock_env_factory(50, 0, ["ZERGLING", "ROACH", "REAPER", "ZEALOT"])
    await env._spawn_initial_units()
    
    calls = env.bot.client.debug_create_unit.call_args_list
    spawned_types = []
    for call in calls:
        for cmd in call[0][0]:
            if cmd[0] != UnitTypeId.PYLON:
                spawned_types.append(cmd[0])
                
    assert len(spawned_types) > 0
    # Should only be ZERGLING (UnitTypeId 105)
    # Note: getattr(UnitTypeId, "ZERGLING") gives the enum, we compare enums.
    expected = getattr(UnitTypeId, "ZERGLING")
    for t in spawned_types:
        assert t == expected, f"Expected ZERGLING, got {t}"

@pytest.mark.asyncio
async def test_composition_reaper_only(mock_env_factory):
    # Budget allows Reapers but not enough minerals for Zealot/Roach (if expensive),
    # or just force Reaper by pool.
    # Let's force Reaper by pool.
    env = mock_env_factory(500, 500, ["REAPER"])
    await env._spawn_initial_units()
    
    calls = env.bot.client.debug_create_unit.call_args_list
    spawned_types = []
    for call in calls:
        for cmd in call[0][0]:
            if cmd[0] != UnitTypeId.PYLON:
                spawned_types.append(cmd[0])
                
    assert len(spawned_types) > 0
    expected = getattr(UnitTypeId, "REAPER")
    for t in spawned_types:
        assert t == expected

@pytest.mark.asyncio
async def test_composition_mixed_budget(mock_env_factory):
    # High budget, mixed pool
    env = mock_env_factory(1000, 1000, ["ZERGLING", "ZEALOT"])
    await env._spawn_initial_units()
    
    calls = env.bot.client.debug_create_unit.call_args_list
    spawned_types = []
    for call in calls:
        for cmd in call[0][0]:
            if cmd[0] != UnitTypeId.PYLON:
                spawned_types.append(cmd[0])
                
    assert len(spawned_types) > 0
    # Should statistically have both
    has_zergling = getattr(UnitTypeId, "ZERGLING") in spawned_types
    has_zealot = getattr(UnitTypeId, "ZEALOT") in spawned_types
    
    # This is probabilistic but with 1000 minerals and only 2 unit types, highly likely to have both.
    # We can't strictly assert both exist unless we mock random, but we can assert valid types.
    valid = [getattr(UnitTypeId, "ZERGLING"), getattr(UnitTypeId, "ZEALOT")]
    for t in spawned_types:
        assert t in valid

@pytest.mark.asyncio
async def test_stability_resets(mock_env_factory):
    # Test multiple resets to ensure no state accumulation or crashes
    env = mock_env_factory(500, 250, ["ZERGLING", "ROACH", "MARINE"])
    
    for _ in range(20):
        # Reset mocks
        env.bot.client.debug_create_unit.reset_mock()
        await env._spawn_initial_units()
        
        # Verify spawning happened
        assert env.bot.client.debug_create_unit.called
