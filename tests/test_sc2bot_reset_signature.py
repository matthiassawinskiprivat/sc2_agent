import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.sc2_bot import SC2Bot

@pytest.mark.asyncio
async def test_sc2bot_reset_signature():
    # Setup mock connection
    mock_conn = MagicMock()
    bot = SC2Bot(conn=mock_conn)
    
    # Mock methods that would be called during reset_game_state
    bot.client = AsyncMock()
    bot._refresh_state = AsyncMock()
    
    # This should FAIL if verify is not accepted
    try:
        await bot.reset_game_state(verify=True)
    except TypeError as e:
        pytest.fail(f"reset_game_state does not accept 'verify' argument: {e}")

@pytest.mark.asyncio
async def test_sc2bot_reset_no_verify():
    # Setup mock connection
    mock_conn = MagicMock()
    bot = SC2Bot(conn=mock_conn)
    
    # Mock methods that would be called during reset_game_state
    bot.client = AsyncMock()
    bot._refresh_state = AsyncMock()
    
    # This should pass (backwards compatibility)
    await bot.reset_game_state()
