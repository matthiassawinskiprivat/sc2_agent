"""Spawn-group combat assignment and stationary command anchors (Level7)."""

from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.level_7_bot import Level7Bot


def _bot_with_topology(training_conf: dict) -> Level7Bot:
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.targets_count = training_conf.get("targets", 2)
    bot.previous_distances = {}
    bot.current_assignments = {}
    bot.agent_system = None
    bot.zergling = None
    bot.units = MagicMock()
    bot.enemy_units = MagicMock()
    return bot


def test_grouped_assignment_ignores_closer_enemy_outside_pack():
    """
    With topology, an ally is only matched to enemies in its current pack, not a nearer
    enemy from another spawn group.
    """
    training_conf = {"targets": 2, "unit_type": "MARINE", "enemy_type": "ZERGLING"}
    bot = _bot_with_topology(training_conf)

    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((10, 10))
    u2 = MagicMock()
    u2.tag = 2
    u2.position = Point2((50, 50))

    # Group 0 at (0,0): e_close is u1's pack but far; e_far_other is wrong pack but very close to u1
    e_pack = MagicMock()
    e_pack.tag = 101
    e_pack.position = Point2((0, 0))
    e_intruder = MagicMock()
    e_intruder.tag = 201
    e_intruder.position = Point2((10.5, 10.5))

    u1.distance_to.side_effect = lambda e: u1.position.distance_to(e.position)
    u2.distance_to.side_effect = lambda e: u2.position.distance_to(e.position)

    mock_allied = MagicMock()
    mock_allied.__iter__.return_value = [u1, u2]
    mock_enemy = MagicMock()
    mock_enemy.__iter__.return_value = [e_pack, e_intruder]
    mock_enemy.exists = True

    def mock_units_callable(unit_type):
        if unit_type == UnitTypeId.MARINE:
            return mock_allied
        return MagicMock()

    def mock_enemy_units_callable(unit_type):
        if unit_type == UnitTypeId.ZERGLING:
            return mock_enemy
        return MagicMock()

    bot.units = MagicMock(side_effect=mock_units_callable)
    bot.enemy_units = MagicMock(side_effect=mock_enemy_units_callable)

    def find_by_tag(tag):
        if tag == 1:
            return u1
        if tag == 2:
            return u2
        return None

    bot.units.find_by_tag.side_effect = find_by_tag

    def find_enemy_by_tag(tag):
        if tag == 101:
            return e_pack
        if tag == 201:
            return e_intruder
        return None

    mock_enemy.find_by_tag.side_effect = find_enemy_by_tag
    bot.get_active_units = MagicMock(return_value=[u1, u2])

    bot._group_anchors = [Point2((0.0, 0.0)), Point2((100.0, 100.0))]
    bot._enemy_tags_by_group = [{101}, {201}]
    bot._ally_primary_group = {1: 0, 2: 1}

    assignment = bot.refresh_combat_assignments()
    assert assignment[1] == 101
    assert assignment[2] == 201


def test_command_anchor_stationary_until_pack_cleared():
    """Objective uses spawn anchor for the active group, not the moving enemy position."""
    training_conf = {"targets": 2, "unit_type": "MARINE", "enemy_type": "ZERGLING"}
    bot = _bot_with_topology(training_conf)

    anchor0 = Point2((5.0, 5.0))
    anchor1 = Point2((90.0, 90.0))
    bot._group_anchors = [anchor0, anchor1]
    bot._enemy_tags_by_group = [{101}, {201}]
    bot._ally_primary_group = {1: 0}

    u1 = MagicMock()
    u1.tag = 1
    u1.position = Point2((20, 20))

    enemy = MagicMock()
    enemy.tag = 101
    enemy.position = Point2((80, 80))

    mock_allied = MagicMock()
    mock_allied.__iter__.return_value = [u1]
    mock_enemy = MagicMock()
    mock_enemy.__iter__.return_value = [enemy]
    mock_enemy.exists = True

    bot.units = MagicMock(
        side_effect=lambda ut: mock_allied if ut == UnitTypeId.MARINE else MagicMock()
    )
    bot.enemy_units = MagicMock(
        side_effect=lambda ut: mock_enemy if ut == UnitTypeId.ZERGLING else MagicMock()
    )
    bot.units.find_by_tag.side_effect = lambda t: u1 if t == 1 else None
    mock_enemy.find_by_tag.side_effect = lambda t: enemy if t == 101 else None
    bot.get_active_units = MagicMock(return_value=[u1])

    g = bot._active_target_group_for_ally(1)
    assert g == 0
    assert bot._group_anchors[g] == anchor0
    assert bot._group_anchors[g] != enemy.position


def test_command_anchor_advances_when_primary_pack_dead():
    training_conf = {"targets": 2, "unit_type": "MARINE", "enemy_type": "ZERGLING"}
    bot = _bot_with_topology(training_conf)

    bot._group_anchors = [Point2((0.0, 0.0)), Point2((50.0, 50.0))]
    bot._enemy_tags_by_group = [{101}, {201}]
    bot._ally_primary_group = {1: 0}

    u1 = MagicMock()
    u1.tag = 1

    e2 = MagicMock()
    e2.tag = 201
    e2.position = Point2((55, 55))

    mock_allied = MagicMock()
    mock_allied.__iter__.return_value = [u1]
    mock_enemy = MagicMock()
    mock_enemy.__iter__.return_value = [e2]
    mock_enemy.exists = True

    bot.units = MagicMock(
        side_effect=lambda ut: mock_allied if ut == UnitTypeId.MARINE else MagicMock()
    )
    bot.enemy_units = MagicMock(
        side_effect=lambda ut: mock_enemy if ut == UnitTypeId.ZERGLING else MagicMock()
    )
    bot.units.find_by_tag.side_effect = lambda t: u1 if t == 1 else None
    mock_enemy.find_by_tag.side_effect = lambda t: e2 if t == 201 else None
    bot.get_active_units = MagicMock(return_value=[u1])

    assert bot._active_target_group_for_ally(1) == 1
    assert bot._group_anchors[1] == Point2((50.0, 50.0))


def test_active_group_wraps_to_earlier_pack_when_forward_chain_cleared():
    """
    Primary slot 1 is cleared first (e.g. ally focused there); pack 0 still has enemies.
    Forward-only scan would return None; wrap must pick group 0 so assignment + anchor recover.
    """
    training_conf = {"targets": 2, "unit_type": "MARINE", "enemy_type": "ZERGLING"}
    bot = _bot_with_topology(training_conf)

    bot._group_anchors = [Point2((0.0, 0.0)), Point2((50.0, 50.0))]
    bot._enemy_tags_by_group = [{101}, set()]
    bot._ally_primary_group = {1: 1}

    e1 = MagicMock()
    e1.tag = 101
    mock_enemy = MagicMock()
    mock_enemy.__iter__.return_value = [e1]
    mock_enemy.exists = True
    mock_enemy.find_by_tag.side_effect = lambda t: e1 if t == 101 else None
    bot.enemy_units = MagicMock(
        side_effect=lambda ut: mock_enemy if ut == UnitTypeId.ZERGLING else MagicMock()
    )

    assert bot._active_target_group_for_ally(1) == 0
    assert bot._allowed_enemy_tags_for_ally_combat(1) == {101}
