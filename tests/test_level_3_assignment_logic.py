import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.level_3_bot import Level3Bot
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from scipy.optimize import linear_sum_assignment

class MockUnit:
    def __init__(self, tag, pos):
        self.tag = tag
        self.position = pos
    def distance_to(self, other):
        return ((self.position[0] - other.position[0])**2 + (self.position[1] - other.position[1])**2)**0.5

def test_optimal_assignment_logic():
    # Setup units and beacons such that greedy would fail but optimal would succeed
    # Unit 1 at (0,0), Unit 2 at (10,0)
    # Beacon A at (2,0), Beacon B at (12,0)
    # Greedy:
    # U1 -> closest is A (dist 2)
    # U2 -> closest is B (dist 2)
    # This is actually optimal too.
    
    # Let's try:
    # U1 at (0,0), U2 at (1,0)
    # B1 at (0, 10), B2 at (1, 1)
    # Greedy:
    # U1 -> B2 (dist 1.41)
    # U2 -> B2 (dist 1.0)
    # If U1 goes first and takes B2, U2 is left with B1 (dist 9+?)
    
    # Better example for linear_sum_assignment:
    # U1 at (0,0), U2 at (10,10)
    # B1 at (0,1), B2 at (10,11)
    # Distance matrix:
    #      B1    B2
    # U1   1    14.14
    # U2  14.14   1
    # Optimal: U1-B1, U2-B2. Total dist = 2
    
    # What if:
    # U1 at (0,0), U2 at (0.1, 0)
    # B1 at (0, 1), B2 at (10, 0)
    # Greedy (if U2 processed first): U2 -> B1 (dist 1), U1 -> B2 (dist 10). Total 11
    # Optimal: U1 -> B1 (dist 1), U2 -> B2 (dist 9.9). Total 10.9
    
    # Let's use the U1(0,0), U2(10,10) example for clarity.
    u1 = MockUnit(1, (0,0))
    u2 = MockUnit(2, (10,10))
    b1 = MockUnit(101, (0,1))
    b2 = MockUnit(102, (10,11))
    
    bot = Level3Bot(conn=None, training_conf={})
    bot.get_active_zerglings = MagicMock(return_value=[u1, u2])
    
    mock_beacons = MagicMock()
    mock_beacons.__iter__.return_value = [b1, b2]
    mock_beacons.amount = 2
    bot.units = MagicMock(return_value=mock_beacons)
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL
    
    # We need to monkeypatch get_optimal_assignments if we haven't implemented it yet,
    # or just test the implementation after we add it.
    # Since I'm doing TDD, I'll implement it now in the bot and then run this.
    
    # For now, let's just test the scipy part to be sure.
    dist_matrix = np.zeros((2, 2))
    dist_matrix[0, 0] = u1.distance_to(b1) # 1
    dist_matrix[0, 1] = u1.distance_to(b2) # ~14.8
    dist_matrix[1, 0] = u2.distance_to(b1) # ~13.4
    dist_matrix[1, 1] = u2.distance_to(b2) # 1
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    assert list(row_ind) == [0, 1]
    assert list(col_ind) == [0, 1] # U1-B1, U2-B2

def test_bot_get_optimal_assignments():
    u1 = MockUnit(1, (0,0))
    u2 = MockUnit(2, (10,10))
    b1 = MockUnit(101, (0,1))
    b2 = MockUnit(102, (10,11))
    
    bot = Level3Bot(conn=None, training_conf={})
    bot.get_active_zerglings = MagicMock(return_value=[u1, u2])
    
    mock_beacons = MagicMock()
    mock_beacons.exists = True
    mock_beacons.__iter__.return_value = [b1, b2]
    mock_beacons.amount = 2
    bot.units = MagicMock(return_value=mock_beacons)
    bot.all_units = MagicMock(return_value=mock_beacons)
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL
    
    assignments = bot.get_optimal_assignments()
    assert assignments == {1: 101, 2: 102}

def test_bot_persistence():
    u1 = MockUnit(1, (0,0))
    u2 = MockUnit(2, (10,10))
    b1 = MockUnit(101, (0,1))
    b2 = MockUnit(102, (10,11))
    
    bot = Level3Bot(conn=None, training_conf={})
    bot.get_active_zerglings = MagicMock(return_value=[u1, u2])
    
    mock_beacons = MagicMock()
    mock_beacons.exists = True
    mock_beacons.__iter__.return_value = [b1, b2]
    mock_beacons.amount = 2
    bot.units = MagicMock(return_value=mock_beacons)
    bot.all_units = MagicMock(return_value=mock_beacons)
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL
    
    # Initialize assignments
    bot.current_assignments = bot.get_optimal_assignments()
    assert bot.current_assignments == {1: 101, 2: 102}
    
    # Move unit 1 closer to beacon 2, but assignment should NOT change
    u1.position = (10, 9) # Now closer to B2(10,11) than B1(0,1)
    
    # get_assigned_distances should still use original assignments
    bot.units.find_by_tag = MagicMock(side_effect=lambda tag: {1:u1, 2:u2, 101:b1, 102:b2}.get(tag))
    bot.all_units.find_by_tag = MagicMock(side_effect=lambda tag: {1:u1, 2:u2, 101:b1, 102:b2}.get(tag))
    
    distances = bot.get_assigned_distances()
    assert 1 in distances
    assert distances[1] == u1.distance_to(b1) # Should be distance to B1, not B2
    assert distances[2] == u2.distance_to(b2)

@pytest.mark.asyncio
async def test_bot_collection_logic():
    from unittest.mock import AsyncMock
    u1 = MockUnit(1, (0,0))
    u2 = MockUnit(2, (10,10))
    b1 = MockUnit(101, (0,0.5)) # Close to U1
    b2 = MockUnit(102, (10,30)) # Far from U2
    
    bot = Level3Bot(conn=None, training_conf={"targets": 2})
    bot.get_active_zerglings = MagicMock(return_value=[u1, u2])
    
    mock_beacons = MagicMock()
    mock_beacons.exists = True
    mock_beacons.__iter__.return_value = [b1, b2]
    mock_beacons.amount = 2
    bot.units = MagicMock(return_value=mock_beacons)
    bot.all_units = MagicMock(return_value=mock_beacons)
    bot.units.find_by_tag = MagicMock(side_effect=lambda tag: {1:u1, 2:u2, 101:b1, 102:b2}.get(tag))
    bot.all_units.find_by_tag = MagicMock(side_effect=lambda tag: {1:u1, 2:u2, 101:b1, 102:b2}.get(tag))
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL
    bot.client = MagicMock()
    bot.client.debug_kill_unit = AsyncMock()
    
    # Initialize assignments
    bot.current_assignments = {1: 101, 2: 102}
    
    # U1 is at (0,0), B1 is at (0, 0.5) -> Distance 0.5 < 2.0. Should collect.
    # U2 is at (10,10), B2 is at (10,30) -> Distance 20 > 2.0. Should NOT collect.
    rewards = await bot.check_collections()
    
    assert 1 in rewards
    assert rewards[1] == 50.0 # 100 / 2
    # Assignment stays until the env step removes it after kill (check_collections buffers only).
    assert 1 in bot.current_assignments
    assert 2 in bot.current_assignments
    bot.client.debug_kill_unit.assert_not_called()

@pytest.mark.asyncio
async def test_bot_no_collection_of_unassigned_beacon():
    from unittest.mock import AsyncMock
    u1 = MockUnit(1, (10, 10.5)) # Close to B2 (10,11)
    u2 = MockUnit(2, (0,0))
    b1 = MockUnit(101, (0,1))
    b2 = MockUnit(102, (10,11))
    
    bot = Level3Bot(conn=None, training_conf={"targets": 2})
    bot.current_assignments = {1: 101, 2: 102} # U1 assigned to B1, U2 assigned to B2
    
    # U1 is at (10, 10.5), B2 is at (10, 11). Dist 0.5.
    # BUT U1 is assigned to B1 (dist ~13).
    # It should NOT collect B2.
    
    bot.units = MagicMock()
    bot.all_units = MagicMock()
    bot.units.find_by_tag = MagicMock(side_effect=lambda tag: {1:u1, 2:u2, 101:b1, 102:b2}.get(tag))
    bot.all_units.find_by_tag = MagicMock(side_effect=lambda tag: {1:u1, 2:u2, 101:b1, 102:b2}.get(tag))
    bot.client = MagicMock()
    bot.client.debug_kill_unit = AsyncMock()
    
    rewards = await bot.check_collections()
    assert rewards == {}
    assert 1 in bot.current_assignments
    assert 2 in bot.current_assignments
    bot.client.debug_kill_unit.assert_not_called()


def test_frozen_assignments_no_rehungarian_when_peer_beacon_missing():
    """Survivors keep their initial beacon; missing peer beacon must not trigger Hungarian."""
    u1 = MockUnit(1, (0, 0))
    u2 = MockUnit(2, (10, 0))
    b2 = MockUnit(102, (10, 10))
    u1.type_id = UnitTypeId.ZERGLING
    u2.type_id = UnitTypeId.ZERGLING
    b2.type_id = UnitTypeId.BEACON_TERRANSMALL

    training_conf = {
        "targets": 2,
        "high_level_command": 0,
        "active_layers": ["terrain"],
        "active_tactical_layers": ["terrain"],
    }
    with patch.object(SC2Bot, "__init__", lambda self, *a, **kw: None):
        bot = Level3Bot.__new__(Level3Bot)
    bot.conn = MagicMock()
    bot.training_conf = training_conf
    bot.targets_count = 2
    bot.allied_unit_type = UnitTypeId.ZERGLING
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL
    bot.previous_distances = {}
    bot.pending_success_rewards = {}
    bot.completed_tags = set()
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {
        "agent_0": np.zeros((1, 128, 128), dtype=np.float32)
    }
    bot.current_assignments = {1: 101, 2: 102}
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL

    def find_by_tag(tag):
        return {1: u1, 2: u2, 102: b2}.get(tag)

    mock_all = MagicMock()
    mock_all.find_by_tag = MagicMock(side_effect=find_by_tag)
    beacons_coll = MagicMock()
    beacons_coll.__iter__ = MagicMock(return_value=iter([b2]))
    beacons_coll.exists = True
    mock_all.__call__ = MagicMock(return_value=beacons_coll)
    bot.all_units = mock_all

    allied = MagicMock()
    allied.__iter__ = MagicMock(return_value=iter([u1, u2]))
    bot.units = MagicMock()
    bot.units.find_by_tag = MagicMock(side_effect=find_by_tag)
    bot.units.filter = MagicMock(return_value=allied)

    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot._game_info = MagicMock()

    def no_hungarian():
        raise AssertionError("get_optimal_assignments must not run mid-episode")

    bot.get_optimal_assignments = no_hungarian

    bot.get_observation_for_unit(2)
    assert bot.current_assignments.get(2) == 102
    raw = bot.agent_system.process_observation.call_args.kwargs["observation"]
    assert raw["command_target"] == [b2]

    bot.get_observation_for_unit(1)
    assert 1 not in bot.current_assignments
    assert bot.current_assignments.get(2) == 102
    raw_u1 = bot.agent_system.process_observation.call_args.kwargs["observation"]
    assert raw_u1["command_target"] == []

