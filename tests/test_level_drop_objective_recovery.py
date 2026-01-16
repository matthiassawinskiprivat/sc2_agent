import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock
from src.level_drop_bot import LevelDropBot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockUnit:
    def __init__(self, tag, pos, type_id=None):
        self.tag = tag
        self.position = pos
        self.type_id = type_id
        self.radius = 1.0
        self.is_structure = True
        self.is_enemy = True

@pytest.mark.asyncio
async def test_level_drop_objective_recovery():
    """Verify that LevelDropBot can find the objective if it's missing at reset."""
    bot = LevelDropBot(conn=MagicMock())
    
    # Mock empty enemy structures initially
    empty_structures = MagicMock()
    empty_structures.filter.return_value = []
    
    # We use a PropertyMock for enemy_structures because it's usually a property in BotAI
    type(bot).enemy_structures = PropertyMock(return_value=empty_structures)
    type(bot).units = PropertyMock(return_value=MagicMock())
    type(bot).enemy_units = PropertyMock(return_value=[])
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.game_loop = 0
    
    # 1. Reset (Should NOT find objective)
    await bot.reset_game_state()
    assert bot._objective_pos is None
    assert bot.command_center_tag is None
    
    # 2. Simulate observation at Step 1 (Still missing)
    # Mocking self.units.find_by_tag for get_observation_for_unit
    dummy_marine = MockUnit(1, Point2((10, 10)), UnitTypeId.MARINE)
    bot.units.find_by_tag.return_value = dummy_marine
    
    # Centering alias used in bot
    bot.zergling = dummy_marine
    
    # Mock AgentSystem
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}
    
    obs, _ = bot.get_observation_for_unit(1)
    call_args = bot.agent_system.process_observation.call_args
    raw_obs = call_args.kwargs['observation']
    assert raw_obs['command_target'] is None
    
    # 3. Step 5: Enemy structure appears
    target_cc = MockUnit(999, Point2((100, 100)), UnitTypeId.COMMANDCENTER)
    found_structures = MagicMock()
    found_structures.filter.return_value = found_structures
    found_structures.first = target_cc
    found_structures.__iter__.return_value = [target_cc]
    found_structures.find_by_tag.return_value = target_cc
    
    # Update enemy_structures mock
    type(bot).enemy_structures = PropertyMock(return_value=found_structures)
    
    # 4. Get observation again (Should find it now)
    obs, _ = bot.get_observation_for_unit(1)
    call_args = bot.agent_system.process_observation.call_args
    raw_obs = call_args.kwargs['observation']
    
    # Should now pass after implementation
    assert raw_obs['command_target'] == Point2((100, 100))
    assert bot._objective_pos == Point2((100, 100))