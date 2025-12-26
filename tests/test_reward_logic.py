
import pytest
from unittest.mock import MagicMock
from src.sc2_bot import SC2Bot

class MockPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockUnit:
    def __init__(self, x, y):
        self.position = MockPosition(x, y)

def test_reward_logic_dense():
    # Setup bot mock
    bot = SC2Bot(conn=MagicMock())
    
    # 1. Initialize previous_distance (mimic _get_observation)
    bot.zergling = MockUnit(10, 10)
    bot.beacon = MockUnit(20, 20)
    
    # Initial distance calculation
    initial_dist = ((10-20)**2 + (10-20)**2)**0.5
    
    # Manually trigger initialization like _get_observation would
    bot.previous_distance = initial_dist
    
    # 2. Move closer
    bot.zergling = MockUnit(12, 12)
    new_dist = ((12-20)**2 + (12-20)**2)**0.5
    
    reward, reached = bot._calculate_reward()
    
    expected_reward = (initial_dist - new_dist) * 1.0
    assert reward == pytest.approx(expected_reward)
    assert not reached
    assert bot.previous_distance == new_dist

def test_reward_logic_win():
    bot = SC2Bot(conn=MagicMock())
    bot.zergling = MockUnit(19, 19)
    bot.beacon = MockUnit(20, 20)
    
    # Target reached (dist < 2.0)
    # dist = (1**2 + 1**2)**0.5 = 1.414 < 2.0
    
    reward, reached = bot._calculate_reward()
    assert reward == 100.0
    assert reached

def test_reward_logic_reset():
    bot = SC2Bot(conn=MagicMock())
    bot.previous_distance = 5.0
    bot.player_id = 1
    
    # Use a dummy client mock for reset_game_state
    from unittest.mock import AsyncMock
    bot.client = AsyncMock()
    bot.units = MagicMock()
    bot.units.exists = False
    
    # Mocking for debug_create_unit
    bot.client.debug_create_unit = AsyncMock()
    bot.client._send_debug = AsyncMock()
    
    # reset_game_state should set previous_distance to None
    # We need to mock a lot of things to call reset_game_state fully,
    # or we just test the specific line.
    # Let's try to mock enough.
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid.width = 100
    bot.game_info.pathing_grid.height = 100
    
    import asyncio
    async def dummy_reset():
        await bot.reset_game_state()
    
    loop = asyncio.new_event_loop()
    loop.run_until_complete(dummy_reset())
    
    assert bot.previous_distance is None
