import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.level_drop_bot import LevelDropBot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.units import Units
from sc2.position import Point2

@pytest.mark.asyncio
async def test_level_drop_bot_no_cleanup_on_reset():
    """
    Verifies that LevelDropBot DOES NOT attempt to kill units during reset.
    """
    # 1. Setup Mock Bot
    conn = MagicMock()
    bot = LevelDropBot(conn=conn, training_conf={})
    
    # Mock Client
    bot.client = MagicMock()
    bot.client.debug_kill_unit = AsyncMock()
    
    # Mock structures attribute
    mock_base = MagicMock()
    mock_base.tag = 456
    mock_base.type_id = UnitTypeId.HATCHERY
    mock_base.position = Point2((80, 80))
    
    from sc2.units import Units
    bot.structures = Units([], bot)
    bot.enemy_structures = Units([mock_base], bot)
    bot.units = Units([], bot)
    bot.enemy_units = Units([], bot)
    
    # 2. Trigger Reset
    await bot.reset_game_state()
    
    # 3. Verify: debug_kill_unit should NEVER be called
    assert bot.client.debug_kill_unit.call_count == 0
    assert bot.command_center_tag == 456

@pytest.mark.asyncio
async def test_level_drop_bot_persistent_objective():
    """
    Verifies that LevelDropBot renders the objective even when not visible.
    """
    conn = MagicMock()
    bot = LevelDropBot(conn=conn, training_conf={})
    
    # 1. Simulate Reset (Identify Objective)
    mock_base = MagicMock()
    mock_base.tag = 456
    mock_base.type_id = UnitTypeId.HATCHERY
    mock_base.position = Point2((80, 80))
    
    from sc2.units import Units
    bot.enemy_structures = Units([mock_base], bot)
    bot.client = AsyncMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    
    await bot.reset_game_state()
    assert bot._objective_pos == Point2((80, 80))
    
    # 2. Simulate Observation (Objective now hidden in FoW)
    bot.enemy_structures = Units([], bot) # Nothing visible
    
    # Mock allied unit to center on
    mock_marine = MagicMock()
    mock_marine.tag = 1
    mock_marine.type_id = UnitTypeId.MARINE
    bot.units = Units([mock_marine], bot)
    
    # Mock Agent System
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": "processed_data"}
    
    # Trigger observation
    obs, _ = bot.get_observation_for_unit(1)
    
    # Verify raw_obs passed to agent_system
    args, kwargs = bot.agent_system.process_observation.call_args
    raw_obs = kwargs["observation"]
    
    assert raw_obs["command_target"] == Point2((80, 80)), "Objective should be persistent!"