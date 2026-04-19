import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock
from src.level_7_bot import Level7Bot
from src.utils.command_constants import CommandConstants
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

def test_greedy_assignment_logic_combat():
    """
    Verify that Level7Bot uses Hungarian assignment (one-to-one when counts match).
    """
    training_conf = {
        "targets": 2,
        "unit_type": "MARINE",
        "enemy_type": "ZERGLING"
    }
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.previous_distances = {}
    bot.current_assignments = {}
    
    # Distances are arranged so optimal one-to-one pairing is:
    # u1 -> e1 and u2 -> e2
    
    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    
    u2 = MagicMock()
    u2.tag = 2
    u2.position = Point2((11, 11))
    
    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((0, 0))
    
    e2 = MagicMock()
    e2.tag = 102
    e2.position = Point2((100, 100))
    u1.distance_to.side_effect = lambda enemy: 1.0 if enemy.tag == 101 else 100.0
    u2.distance_to.side_effect = lambda enemy: 100.0 if enemy.tag == 101 else 1.0
    
    mock_units_allied = MagicMock()
    mock_units_allied.__iter__.return_value = [u1, u2]
    
    mock_units_enemy = MagicMock()
    mock_units_enemy.__iter__.return_value = [e1, e2]
    mock_units_enemy.exists = True

    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.MARINE:
            return mock_units_allied
        return MagicMock()
        
    bot.units = MagicMock(side_effect=mock_units_callable)
    
    def mock_enemy_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_enemy
        return MagicMock()
    bot.enemy_units = MagicMock(side_effect=mock_enemy_units_callable)
    
    def find_by_tag(tag):
        if tag == 1: return u1
        if tag == 2: return u2
        return None
    bot.units.find_by_tag.side_effect = find_by_tag
    
    def find_enemy_by_tag(tag):
        if tag == 101: return e1
        if tag == 102: return e2
        return None
    mock_units_enemy.find_by_tag.side_effect = find_enemy_by_tag

    bot.get_active_units = MagicMock(return_value=[u1, u2])

    assignment = bot.get_closest_targets_combat()
    
    assert assignment[1] == 101
    assert assignment[2] == 102

def test_greedy_reward_calculation_combat():
    """
    Verify distance reward in Level 7 uses average change in distance to closest enemies.
    """
    training_conf = {
        "targets": 2,
        "unit_type": "MARINE",
        "enemy_type": "ZERGLING",
        "rewards": {
            "distance_multiplier": 1.0,
            "step_penalty": 0.0
        }
    }
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.previous_distances = {}
    bot.current_assignments = {}
    
    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    u1.ground_range = 5.0
    u1.air_range = 0.0
    u1.is_flying = False
    u1.distance_to.side_effect = lambda other: u1.position.distance_to(other.position)

    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((0, 0))
    e1.ground_range = 1.0
    e1.air_range = 0.0
    e1.is_flying = False
    
    mock_units_allied = MagicMock()
    mock_units_allied.__iter__.return_value = [u1]
    
    mock_units_enemy = MagicMock()
    mock_units_enemy.__iter__.return_value = [e1]
    mock_units_enemy.closest_to.side_effect = lambda unit: e1
    mock_units_enemy.exists = True

    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.MARINE:
            return mock_units_allied
        return MagicMock()
    bot.units = MagicMock(side_effect=mock_units_callable)
    bot.units.find_by_tag.side_effect = lambda tag: u1 if tag == 1 else None
    
    def mock_enemy_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_units_enemy
        return MagicMock()
    bot.enemy_units = MagicMock(side_effect=mock_enemy_units_callable)
    mock_units_enemy.find_by_tag.side_effect = lambda tag: e1 if tag == 101 else None
    
    bot.get_active_units = MagicMock(return_value=[u1])
    
    # Initial distance is sqrt(10^2 + 10^2) = 14.142
    bot.previous_distances = {1: 14.142}
    
    # Move unit closer to (5, 5) -> dist = 7.071
    u1.position = Point2((5, 5))
    
    reward = bot.calculate_distance_reward()
    
    assert pytest.approx(reward, 0.01) == 7.071


def test_enemy_target_fallback_when_greedy_assignment_misses_unit():
    """If greedy mapping misses a unit, observation still gets a nearest live enemy target."""
    bot = object.__new__(Level7Bot)
    bot.training_conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.zergling = None

    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    u1.distance_to.side_effect = lambda other: u1.position.distance_to(other.position)

    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((20, 20))

    enemy_collection = MagicMock()
    enemy_collection.exists = True
    enemy_collection.__iter__.return_value = [e1]
    enemy_collection.find_by_tag.return_value = e1

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = u1
    bot.get_active_units = MagicMock(return_value=[u1])
    bot.enemy_units = MagicMock(return_value=enemy_collection)
    bot.enemy_structures = []
    bot.structures = []
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot._game_info = MagicMock()
    bot._game_data = None
    bot.get_closest_targets_combat = MagicMock(return_value={})
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": {"map": np.zeros((1, 128, 128), dtype=np.float32)}}

    bot.get_observation_for_unit(u1.tag)

    raw_obs = bot.agent_system.process_observation.call_args.kwargs["observation"]
    assert raw_obs["command_target"] == e1.position
    assert raw_obs["high_level_command"] == CommandConstants.SCOUT


def test_universal_observation_neutralizes_yaml_attack_directive():
    """YAML may say attack for curriculum naming; universal micro obs must not fix command_vector to ATTACK."""
    bot = object.__new__(Level7Bot)
    bot.training_conf = {
        "agent_architecture": "universal",
        "high_level_command": "attack",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.zergling = None

    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    u1.distance_to.side_effect = lambda other: u1.position.distance_to(other.position)

    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((20, 20))

    enemy_collection = MagicMock()
    enemy_collection.exists = True
    enemy_collection.__iter__.return_value = [e1]
    enemy_collection.find_by_tag.return_value = e1

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = u1
    bot.get_active_units = MagicMock(return_value=[u1])
    bot.enemy_units = MagicMock(return_value=enemy_collection)
    bot.enemy_structures = []
    bot.structures = []
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot._game_info = MagicMock()
    bot._game_data = None
    bot.get_closest_targets_combat = MagicMock(return_value={})
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": {"map": np.zeros((1, 128, 128), dtype=np.float32)}}

    bot.get_observation_for_unit(u1.tag)

    raw_obs = bot.agent_system.process_observation.call_args.kwargs["observation"]
    assert raw_obs["high_level_command"] == CommandConstants.SCOUT


def test_original_arch_passes_through_high_level_command():
    bot = object.__new__(Level7Bot)
    bot.training_conf = {
        "agent_architecture": "original",
        "high_level_command": "attack",
        "active_layers": ["allied_unit_density"],
    }
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.zergling = None

    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    e1 = MagicMock()
    e1.tag = 101
    e1.position = Point2((20, 20))

    enemy_collection = MagicMock()
    enemy_collection.exists = True
    enemy_collection.__iter__.return_value = [e1]

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = u1
    bot.get_active_units = MagicMock(return_value=[u1])
    bot.enemy_units = MagicMock(return_value=enemy_collection)
    bot.enemy_structures = []
    bot.structures = []
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot._game_info = MagicMock()
    bot._game_data = None
    bot.get_closest_targets_combat = MagicMock(return_value={})
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 128, 128), dtype=np.float32)}

    bot.get_observation_for_unit(u1.tag)

    raw_obs = bot.agent_system.process_observation.call_args.kwargs["observation"]
    assert raw_obs["high_level_command"] == "attack"


def test_refresh_combat_assignments_covers_all_active_allies():
    bot = object.__new__(Level7Bot)
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.current_assignments = {}

    u1 = MagicMock(tag=1)
    u2 = MagicMock(tag=2)
    u1.distance_to.side_effect = lambda other: 1.0 if other.tag == 101 else 10.0
    u2.distance_to.side_effect = lambda other: 2.0 if other.tag == 102 else 10.0

    e1 = MagicMock(tag=101)
    e2 = MagicMock(tag=102)

    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    enemies.closest_to.side_effect = lambda ally: e1 if ally.tag == 1 else e2

    bot.get_active_units = MagicMock(return_value=[u1, u2])
    bot.enemy_units = MagicMock(return_value=enemies)

    assignments = bot.refresh_combat_assignments()
    assert set(assignments.keys()) == {1, 2}
    assert assignments[1] == 101
    assert assignments[2] == 102


def test_refresh_combat_assignments_retargets_after_enemy_death():
    bot = object.__new__(Level7Bot)
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.current_assignments = {1: 101, 2: 102}

    u1 = MagicMock(tag=1)
    u2 = MagicMock(tag=2)
    u1.distance_to.side_effect = lambda other: 3.0
    u2.distance_to.side_effect = lambda other: 4.0

    e2 = MagicMock(tag=102)
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e2]  # enemy 101 died
    enemies.closest_to.return_value = e2

    bot.get_active_units = MagicMock(return_value=[u1, u2])
    bot.enemy_units = MagicMock(return_value=enemies)

    assignments = bot.refresh_combat_assignments()
    assert assignments == {1: 102, 2: 102}


def test_refresh_combat_assignments_keeps_existing_valid_pairs():
    bot = object.__new__(Level7Bot)
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.current_assignments = {1: 101, 2: 102}

    u1 = MagicMock(tag=1)
    u2 = MagicMock(tag=2)
    # Distances now prefer cross-pairing, but valid existing pairs should be preserved.
    u1.distance_to.side_effect = lambda enemy: 1.0 if enemy.tag == 102 else 5.0
    u2.distance_to.side_effect = lambda enemy: 1.0 if enemy.tag == 101 else 5.0

    e1 = MagicMock(tag=101)
    e2 = MagicMock(tag=102)
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]

    bot.get_active_units = MagicMock(return_value=[u1, u2])
    bot.enemy_units = MagicMock(return_value=enemies)

    assignments = bot.refresh_combat_assignments()
    assert assignments == {1: 101, 2: 102}


def test_queue_idle_combat_orders_only_for_idle_assigned_units():
    bot = object.__new__(Level7Bot)
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.current_assignments = {1: 101, 2: 102}

    u1 = MagicMock(tag=1)
    u1.orders = []  # idle => should receive keepalive order
    u2 = MagicMock(tag=2)
    u2.orders = [MagicMock()]  # busy => must not be overridden

    e1 = MagicMock(tag=101)
    e2 = MagicMock(tag=102)
    enemies = MagicMock()
    enemies.exists = True
    enemies.find_by_tag.side_effect = lambda tag: e1 if tag == 101 else (e2 if tag == 102 else None)

    bot.refresh_combat_assignments = MagicMock(return_value={1: 101, 2: 102})
    bot.get_active_units = MagicMock(return_value=[u1, u2])
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.do = MagicMock()

    queued = bot.queue_idle_combat_orders()

    assert queued == 1
    u1.attack.assert_called_once_with(e1)
    u2.attack.assert_not_called()


def test_apply_action_attack_uses_agent_world_target_not_assignment():
    training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_local", "attack_local"],
    }
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.action_heads = training_conf["action_heads"]
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (30.0, 30.0)

    unit = MagicMock()
    unit.tag = 1
    unit.position = Point2((10, 10))
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = unit

    assigned_enemy = MagicMock()
    assigned_enemy.tag = 101
    assigned_enemy.position = Point2((20, 20))
    enemies = MagicMock()
    enemies.exists = True
    enemies.find_by_tag.return_value = assigned_enemy
    enemies.__iter__.return_value = [assigned_enemy]
    bot.enemy_units = MagicMock(return_value=enemies)

    bot.refresh_combat_assignments = MagicMock(return_value={1: 101})
    bot.do = MagicMock()
    bot.last_action_was_attack_by_unit = {}
    bot.last_attack_target_tag_by_unit = {}
    bot._kite_damage_credit_ok = {}
    bot._combat_reward_suppressed = {}

    # attack_local head index = 1
    asyncio.run(bot.apply_action(1, np.array([1, 0], dtype=np.int64)))

    unit.attack.assert_called_once_with(Point2((30.0, 30.0)))
    bot.refresh_combat_assignments.assert_not_called()
