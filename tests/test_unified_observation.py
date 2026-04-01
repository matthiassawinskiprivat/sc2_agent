import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from src.sc2_bot import SC2Bot


class MockUnit:
    def __init__(self, tag, pos, type_id=None):
        self.tag = tag
        self.position = pos
        self.type_id = type_id
        self.radius = 1.0
        self.is_structure = False
        self.is_enemy = True


def _mock_units_result(units_list):
    """Create a mock that behaves like a burnysc2 Units collection."""
    m = MagicMock()
    m.exists = len(units_list) > 0
    m.first = units_list[0] if units_list else None
    m.__iter__ = MagicMock(return_value=iter(units_list))
    m.__len__ = MagicMock(return_value=len(units_list))
    m.closest_to = MagicMock(return_value=units_list[0] if units_list else None)
    m.find_by_tag = MagicMock(
        side_effect=lambda tag: next((u for u in units_list if u.tag == tag), None)
    )
    # filter returns a new result based on the predicate
    def _filter(pred):
        return _mock_units_result([u for u in units_list if pred(u)])
    m.filter = MagicMock(side_effect=_filter)
    return m


def _create_bot(**training_conf_overrides):
    """Instantiate SC2Bot bypassing BotAI internals."""
    with patch.object(SC2Bot, "__init__", lambda self, *a, **kw: None):
        bot = SC2Bot.__new__(SC2Bot)
    bot.conn = MagicMock()
    bot.training_conf = training_conf_overrides
    bot.agent_system = None
    bot.zergling = None
    bot.beacon = None
    bot.previous_distance = None
    bot.last_target_pos = None
    bot._game_state_initialized = False
    bot.action_heads = ["move"]
    bot.categorical_heads = []
    bot.unit_command_uses_self_do = False
    bot.actions = []
    bot.unit_tags_received_action = set()
    return bot


def test_unified_observation_guidance():
    """Verify that SC2Bot uses the beacon position as command_target."""
    bot = _create_bot()
    beacon = MockUnit(201, Point2((20, 20)), UnitTypeId.BEACON_TERRANSMALL)

    def mock_units_call(type_id):
        return _mock_units_result([])

    bot.units = MagicMock(side_effect=mock_units_call)
    bot.units.filter = MagicMock(return_value=_mock_units_result([]))
    bot.all_units = _mock_units_result([beacon])
    bot.enemy_units = _mock_units_result([])
    bot.structures = _mock_units_result([])
    bot.enemy_structures = _mock_units_result([])

    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None

    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}

    bot._get_observation()

    call_args = bot.agent_system.process_observation.call_args
    assert call_args is not None
    raw_obs = call_args.kwargs["observation"]
    assert raw_obs["command_target"] == beacon.position
    assert raw_obs["command_target"] == Point2((20, 20))


def test_unified_observation_fallback():
    """Verify fallback if no beacon exists."""
    bot = _create_bot()
    bot.last_target_pos = Point2((5, 5))

    def mock_units_call(type_id):
        return _mock_units_result([])

    bot.units = MagicMock(side_effect=mock_units_call)
    bot.units.filter = MagicMock(return_value=_mock_units_result([]))
    bot.all_units = _mock_units_result([])
    bot.enemy_units = _mock_units_result([])
    bot.structures = _mock_units_result([])
    bot.enemy_structures = _mock_units_result([])

    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None

    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}

    bot._get_observation()

    call_args = bot.agent_system.process_observation.call_args
    raw_obs = call_args.kwargs["observation"]
    assert raw_obs["command_target"] == Point2((5, 5))
