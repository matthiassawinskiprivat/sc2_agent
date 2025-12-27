import pytest
from unittest.mock import MagicMock
from src.sc2_bot import SC2Bot
from sc2.data import Race

def test_sc2bot_has_enemy_race():
    """Verify that SC2Bot has the enemy_race attribute after initialization."""
    # Mocking the multiprocessing connection
    conn = MagicMock()
    bot = SC2Bot(conn=conn)
    
    # This should not raise AttributeError
    assert hasattr(bot, "_enemy_race")
    assert bot._enemy_race == Race.Random
