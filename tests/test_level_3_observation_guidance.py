import pytest
import numpy as np
from unittest.mock import MagicMock
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockUnit:
    def __init__(self, tag, pos, type_id=None):
        self.tag = tag
        self.position = pos
        self.type_id = type_id
        self.radius = 1.0
        self.is_structure = False
        self.is_enemy = True

def test_level_3_observation_guidance():
    """Verify that observation includes all beacons and correct guidance target."""
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 2})
    
    # 2 Units, 2 Beacons
    # U1 (0,0) -> B1 (0,10)
    # U2 (10,0) -> B2 (10,10)
    u1 = MockUnit(101, Point2((0, 0)), UnitTypeId.ZERGLING)
    u2 = MockUnit(102, Point2((10, 0)), UnitTypeId.ZERGLING)
    b1 = MockUnit(201, Point2((0, 10)), UnitTypeId.BEACON_TERRANSMALL)
    b2 = MockUnit(202, Point2((10, 10)), UnitTypeId.BEACON_TERRANSMALL)
    
    units = [u1, u2]
    beacons = [b1, b2]
    all_things = units + beacons
    
    # Mock bot.units
    def mock_filter(type_id):
        res = units if type_id == UnitTypeId.ZERGLING else beacons
        m = MagicMock()
        m.__iter__.return_value = res
        m.exists = True
        m.closest_to.return_value = res[0] # Dummy
        return m
    
    bot.units = MagicMock(side_effect=mock_filter)
    bot.units.find_by_tag.side_effect = lambda tag: next((u for u in all_things if u.tag == tag), None)
    
    # Mock state
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    
    # Mock AgentSystem to capture raw_obs
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}
    
    # Mock assignments to be deterministic and decoupled from math utils
    bot._calculate_optimal_assignment = MagicMock(return_value={
        u1.tag: b1.tag,
        u2.tag: b2.tag
    })
    
    # Get observation for U1
    bot.get_observation_for_unit(u1.tag)
    
    # Check what was passed to process_observation
    call_args = bot.agent_system.process_observation.call_args
    assert call_args is not None
    raw_obs = call_args.kwargs['observation']
    
    # 1. Verify "Unblinded": All beacons should be in enemy_units
    assert len(raw_obs['enemy_units']) == 2
    assert b1 in raw_obs['enemy_units']
    assert b2 in raw_obs['enemy_units']
    
    # 2. Verify "Guidance": command_target should be B1's position (0, 10)
    assert raw_obs['command_target'] == b1.position
    
    # Get observation for U2
    bot.get_observation_for_unit(u2.tag)
    call_args = bot.agent_system.process_observation.call_args
    raw_obs = call_args.kwargs['observation']
    
    # Verify Guidance for U2 -> B2 (10, 10)
    assert raw_obs['command_target'] == b2.position

