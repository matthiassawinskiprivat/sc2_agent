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
async def test_level_drop_spatial_propagation():
    """Verify that command_target in LevelDropBot is passed to the AgentSystem."""
    bot = LevelDropBot(conn=MagicMock())
    
    # 1. Setup objective
    target_cc = MockUnit(999, Point2((100, 100)), UnitTypeId.COMMANDCENTER)
    found_structures = MagicMock()
    found_structures.filter.return_value = found_structures
    found_structures.first = target_cc
    found_structures.__iter__.return_value = [target_cc]
    found_structures.find_by_tag.return_value = target_cc
    
    type(bot).enemy_structures = PropertyMock(return_value=found_structures)
    type(bot).units = PropertyMock(return_value=MagicMock())
    type(bot).enemy_units = PropertyMock(return_value=[])
    
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.game_loop = 0
    
    # 2. Reset (Should find objective)
    await bot.reset_game_state()
    assert bot._objective_pos == Point2((100, 100))
    
    # 3. Get observation for a unit
    dummy_marine = MockUnit(1, Point2((10, 10)), UnitTypeId.MARINE)
    bot.units.find_by_tag.return_value = dummy_marine
    bot.zergling = dummy_marine
    
    # Mock AgentSystem to capture input
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}
    
    bot.get_observation_for_unit(1)
    
    # 4. Verify propagation
    call_args = bot.agent_system.process_observation.call_args
    assert call_args is not None
    raw_obs = call_args.kwargs['observation']
    
    assert raw_obs['command_target'] == Point2((100, 100)), "Objective position must be passed as command_target"
