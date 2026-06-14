"""Regression: distance shaping must not penalize units before their first action."""
from unittest.mock import MagicMock

from src.level_3_bot import Level3Bot


def test_first_action_has_no_distance_baseline():
    """Units later in the cycle must not get negative shaping from a reset-time baseline."""
    bot = Level3Bot(conn=MagicMock(), training_conf={"agent_architecture": "universal", "rewards": {}})
    unit = MagicMock()
    unit.tag = 200
    beacon = MagicMock()
    beacon.tag = 900

    bot.current_assignments = {200: 900}
    bot._find_allied_unit_by_tag = MagicMock(return_value=unit)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag = MagicMock(return_value=beacon)

    # Unit was pushed 3 gu away before its first policy turn; no baseline yet.
    unit.distance_to.return_value = 12.0

    reward = bot.calculate_distance_reward(200)

    assert reward == 0.0
    assert bot.previous_distances[200] == 12.0


def test_second_action_uses_prior_baseline():
    bot = Level3Bot(conn=MagicMock(), training_conf={"agent_architecture": "universal", "rewards": {"distance_multiplier": 1.0}})
    unit = MagicMock()
    unit.tag = 200
    beacon = MagicMock()
    beacon.tag = 900

    bot.current_assignments = {200: 900}
    bot._find_allied_unit_by_tag = MagicMock(return_value=unit)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag = MagicMock(return_value=beacon)
    bot.previous_distances[200] = 15.0
    unit.distance_to.return_value = 10.0

    reward = bot.calculate_distance_reward(200)

    assert reward == 5.0
    assert bot.previous_distances[200] == 10.0
