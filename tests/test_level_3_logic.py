import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def test_collection_logic():
    asyncio.run(_test_collection_logic_impl())

async def _test_collection_logic_impl():
    # Setup
    training_conf = {
        "targets": 2,
        "rewards": {"success": 10.0}
    }
    bot = Level3Bot(conn=MagicMock(), training_conf=training_conf)
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    
    # Mock Units
    z1 = MagicMock()
    z1.tag = 101
    z1.position = Point2((10, 10))
    # Implement distance_to logic or mock it
    def z1_dist(other):
        return z1.position.distance_to(other.position)
    z1.distance_to.side_effect = z1_dist
    
    b1 = MagicMock()
    b1.tag = 201
    b1.position = Point2((10, 10)) # Same position -> Collision
    
    z2 = MagicMock()
    z2.tag = 102
    z2.position = Point2((50, 50))
    def z2_dist(other):
        return z2.position.distance_to(other.position)
    z2.distance_to.side_effect = z2_dist
    
    b2 = MagicMock()
    b2.tag = 202
    b2.position = Point2((60, 60)) # Far away
    
    # Mock units collection
    # We need a way for bot.units(TYPE) to return these
    
    # Using a simpler approach: we can manually populate lists if the bot uses them, 
    # but the bot likely queries self.units.
    # We need to mock self.units(TYPE)
    
    def units_side_effect(unit_type):
        mock_collection = MagicMock()
        if unit_type == UnitTypeId.ZERGLING:
            mock_collection.__iter__.return_value = [z1, z2]
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            mock_collection.__iter__.return_value = [b1, b2]
        return mock_collection
        
    bot.units = MagicMock()
    bot.units.side_effect = units_side_effect
    
    # Run logic
    reward = await bot.check_collections()
    
    # Verify
    # b1 and z1 should be killed
    assert bot.client.debug_kill_unit.called
    kill_calls = bot.client.debug_kill_unit.call_args_list
    killed_tags = []
    for c in kill_calls:
        killed_tags.extend(c[0][0])
        
    assert 101 in killed_tags
    assert 201 in killed_tags
    assert 102 not in killed_tags
    assert 202 not in killed_tags
    
    # Verify reward
    # Success reward is 10.0 / 2 targets = 5.0
    assert reward == 5.0
