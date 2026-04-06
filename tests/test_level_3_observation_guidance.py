import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from src.sc2_bot import SC2Bot
from src.level_3_bot import Level3Bot


class MockUnit:
    def __init__(self, tag, pos, type_id=None):
        self.tag = tag
        self.position = pos
        self.type_id = type_id
        self.radius = 1.0
        self.is_structure = False
        self.is_enemy = True


def _mock_units_result(units_list):
    m = MagicMock()
    m.exists = len(units_list) > 0
    m.first = units_list[0] if units_list else None
    m.__iter__ = MagicMock(return_value=iter(units_list))
    m.__len__ = MagicMock(return_value=len(units_list))
    m.closest_to = MagicMock(return_value=units_list[0] if units_list else None)
    m.find_by_tag = MagicMock(
        side_effect=lambda tag: next((u for u in units_list if u.tag == tag), None)
    )
    def _filter(pred):
        return _mock_units_result([u for u in units_list if pred(u)])
    m.filter = MagicMock(side_effect=_filter)
    # Support units(type_id) call syntax (burnysc2 Units.__call__)
    m.side_effect = lambda tid: _mock_units_result([u for u in units_list if u.type_id == tid])
    return m


def _create_level3_bot(**conf):
    """Instantiate Level3Bot bypassing BotAI internals."""
    training_conf = {"targets": 2, **conf}
    with patch.object(SC2Bot, "__init__", lambda self, *a, **kw: None):
        bot = Level3Bot.__new__(Level3Bot)
    bot.conn = MagicMock()
    bot.training_conf = training_conf
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
    bot.targets_count = training_conf.get("targets", 1)
    bot.allied_unit_type = UnitTypeId.ZERGLING
    bot.beacon_type = UnitTypeId.BEACON_TERRANSMALL
    bot.current_assignments = {}
    bot.current_unit_index = 0
    bot.active_unit_tag = None
    bot._unit_rewards = {}
    bot._unit_observations = {}
    return bot


def test_level_3_observation_guidance():
    """Verify that observation includes correct guidance target per unit."""
    bot = _create_level3_bot()

    u1 = MockUnit(101, Point2((0, 0)), UnitTypeId.ZERGLING)
    u2 = MockUnit(102, Point2((10, 0)), UnitTypeId.ZERGLING)
    b1 = MockUnit(201, Point2((0, 10)), UnitTypeId.BEACON_TERRANSMALL)
    b2 = MockUnit(202, Point2((10, 10)), UnitTypeId.BEACON_TERRANSMALL)

    all_things = [u1, u2, b1, b2]

    bot.all_units = _mock_units_result(all_things)
    bot.units = _mock_units_result([u1, u2])
    bot.enemy_units = _mock_units_result([])
    bot.structures = _mock_units_result([])
    bot.enemy_structures = _mock_units_result([])

    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None

    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}

    # Same as after reset_game_state: assignments are set once per episode, not via lazy init.
    bot.current_assignments = {u1.tag: b1.tag, u2.tag: b2.tag}

    # Observation for U1
    bot.get_observation_for_unit(u1.tag)
    call_args = bot.agent_system.process_observation.call_args
    assert call_args is not None
    raw_obs = call_args.kwargs["observation"]

    # enemy_units contains ALL beacons (for density layer rendering)
    assert len(raw_obs["enemy_units"]) == 2
    assert b1 in raw_obs["enemy_units"]
    assert b2 in raw_obs["enemy_units"]
    # command_target is the ASSIGNED beacon for U1
    assert raw_obs["command_target"] == [b1]

    # Observation for U2
    bot.get_observation_for_unit(u2.tag)
    call_args = bot.agent_system.process_observation.call_args
    raw_obs = call_args.kwargs["observation"]

    assert len(raw_obs["enemy_units"]) == 2
    # command_target is the ASSIGNED beacon for U2
    assert raw_obs["command_target"] == [b2]
