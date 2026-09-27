"""Direct unit-tag actuation for the ``snipe`` enemy-pointer command.

Ghost Steady Targeting, Battlecruiser Yamato, Infestor Neural Parasite, and
High Templar Feedback all share one head: actors + enemy tag → per-type ability.
"""
import asyncio
from unittest.mock import MagicMock

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot, UNIT_TARGET_COMMAND_HEADS

ACTION_HEADS = [
    "move_global",
    "attack_local",
    "attack_unit",
    "snipe",
]
CMD_SNIPE = ACTION_HEADS.index("snipe")


def _make_unit(
    tag,
    type_id,
    x=50.0,
    y=50.0,
    owner_id=1,
    *,
    is_biological=False,
    energy=200.0,
    energy_max=200.0,
    is_structure=False,
):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.is_biological = is_biological
    u.is_structure = is_structure
    u.energy = energy
    u.energy_max = energy_max
    u.return_value = object()
    return u


def _bot(*units):
    bot = MagicMock(spec=CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "squad",
        "action_heads": list(ACTION_HEADS),
        "categorical_heads": [],
    }
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot._command_attempt_tags = {
        "move": set(),
        "attack": set(),
        "snipe": set(),
        "aoe": set(),
        "cloak": set(),
        "uncloak": set(),
        "load": set(),
        "unload": set(),
    }
    bot.game_info = None

    by_tag = {u.tag: u for u in units}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get
    bot.units = list(units)

    bot._apply_unit_target_command = (
        lambda selected_tags, *, head_name, target_unit_tag: (
            CaptureTheFlagBot._apply_unit_target_command(
                bot,
                selected_tags,
                head_name=head_name,
                target_unit_tag=target_unit_tag,
            )
        )
    )
    bot._mark_command_attempt = lambda u, kind: CaptureTheFlagBot._mark_command_attempt(
        bot, u, kind
    )
    bot._set_last_order_flags = (
        lambda unit_tag, *, moved=False, attacked=False: (
            CaptureTheFlagBot._set_last_order_flags(
                bot, unit_tag, moved=moved, attacked=attacked
            )
        )
    )

    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (0.0, 0.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot, agent


def _apply(bot, *, selected, target_unit_tag, spatial_idx=100):
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=list(selected),
            anchor_tag=selected[0],
            command_idx=CMD_SNIPE,
            spatial_idx=spatial_idx,
            perspective_player_id=1,
            target_unit_tag=target_unit_tag,
        )
    )


def test_snipe_is_a_unit_target_head():
    assert "snipe" in UNIT_TARGET_COMMAND_HEADS


def test_ghost_snipe_issues_steady_targeting_on_selected_tag():
    ghost = _make_unit(1, UnitTypeId.GHOST, owner_id=1)
    far_bio = _make_unit(2, UnitTypeId.ZERGLING, 99.0, 99.0, owner_id=2, is_biological=True)
    near_bio = _make_unit(3, UnitTypeId.HYDRALISK, 50.5, 50.0, owner_id=2, is_biological=True)
    bot, agent = _bot(ghost, far_bio, near_bio)

    _apply(bot, selected=[1], target_unit_tag=2)

    ghost.assert_called_once_with(AbilityId.EFFECT_GHOSTSNIPE, far_bio)
    bot.do.assert_called_once()
    agent.convert_spatial_action_to_world.assert_not_called()
    assert 1 in bot._command_attempt_tags["snipe"]


def test_ghost_snipe_skips_non_biological_target():
    ghost = _make_unit(1, UnitTypeId.GHOST, owner_id=1)
    stalker = _make_unit(2, UnitTypeId.STALKER, owner_id=2, is_biological=False)
    bot, _ = _bot(ghost, stalker)

    _apply(bot, selected=[1], target_unit_tag=2)

    ghost.assert_not_called()
    bot.do.assert_not_called()
    assert 1 in bot._command_attempt_tags["snipe"]


def test_yamato_issues_on_selected_tag():
    bc = _make_unit(1, UnitTypeId.BATTLECRUISER, owner_id=1)
    corruptor = _make_unit(2, UnitTypeId.CORRUPTOR, owner_id=2)
    bot, _ = _bot(bc, corruptor)

    _apply(bot, selected=[1], target_unit_tag=2)

    bc.assert_called_once_with(AbilityId.YAMATO_YAMATOGUN, corruptor)
    bot.do.assert_called_once()


def test_neural_parasite_issues_on_selected_tag():
    infestor = _make_unit(1, UnitTypeId.INFESTOR, owner_id=1)
    marauder = _make_unit(2, UnitTypeId.MARAUDER, owner_id=2, is_biological=True)
    bot, _ = _bot(infestor, marauder)

    _apply(bot, selected=[1], target_unit_tag=2)

    infestor.assert_called_once_with(
        AbilityId.NEURALPARASITE_NEURALPARASITE, marauder
    )


def test_feedback_issues_on_energy_unit():
    ht = _make_unit(1, UnitTypeId.HIGHTEMPLAR, owner_id=1)
    ghost = _make_unit(2, UnitTypeId.GHOST, owner_id=2, energy_max=200.0)
    bot, _ = _bot(ht, ghost)

    _apply(bot, selected=[1], target_unit_tag=2)

    ht.assert_called_once_with(AbilityId.FEEDBACK_FEEDBACK, ghost)


def test_snipe_never_friendly_fires():
    ghost = _make_unit(1, UnitTypeId.GHOST, owner_id=1)
    ally = _make_unit(2, UnitTypeId.MARINE, owner_id=1, is_biological=True)
    bot, _ = _bot(ghost, ally)

    _apply(bot, selected=[1], target_unit_tag=2)

    ghost.assert_not_called()
    bot.do.assert_not_called()
