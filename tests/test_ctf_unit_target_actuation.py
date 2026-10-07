"""Direct unit-tag actuation for the AlphaStar-style pointer commands.

``attack_unit``, ``heal_unit``, and ``load_unit`` take the tag the policy
selected through the terminal target slot, so there is no cell geometry to miss
and no hidden retargeting. These tests pin that the order goes to exactly that
tag, and that per-caster legality (biology, ownership, cargo fit) still applies.
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
    "heal_unit",
    "load_unit",
]
CMD_ATTACK_UNIT = ACTION_HEADS.index("attack_unit")
CMD_HEAL_UNIT = ACTION_HEADS.index("heal_unit")
CMD_LOAD_UNIT = ACTION_HEADS.index("load_unit")


def _make_unit(
    tag,
    type_id,
    x=50.0,
    y=50.0,
    owner_id=1,
    health=50.0,
    health_max=100.0,
    is_biological=False,
    is_mechanical=False,
    is_structure=False,
    cargo_max=0,
    cargo_used=0,
    cargo_size=1,
):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.health = health
    u.health_max = health_max
    u.radius = 0.5
    u.is_structure = is_structure
    u.is_idle = False
    u.is_biological = is_biological
    u.is_mechanical = is_mechanical
    u.cargo_max = cargo_max
    u.cargo_used = cargo_used
    u.cargo_size = cargo_size
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
    bot._heal_cast_tags = set()
    bot._heal_order_tags = set()
    bot._heal_attempt_tags = set()
    bot._command_attempt_tags = {
        "move": set(),
        "attack": set(),
        "cloak": set(),
        "uncloak": set(),
    }
    bot._heal_lock = {}
    bot._snipe_lock = {}
    bot.game_info = None
    bot.state = MagicMock()
    bot.state.game_loop = 0
    bot.frame_count = 0

    by_tag = {u.tag: u for u in units}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get
    bot.units = list(units)
    bot.enemy_units = []

    bot._apply_unit_target_command = (
        lambda selected_tags, *, head_name, target_unit_tag, perspective_player_id=None: (
            CaptureTheFlagBot._apply_unit_target_command(
                bot,
                selected_tags,
                head_name=head_name,
                target_unit_tag=target_unit_tag,
                perspective_player_id=perspective_player_id,
            )
        )
    )
    bot._refresh_snipe_lock = lambda u: CaptureTheFlagBot._refresh_snipe_lock(bot, u)
    bot._refresh_aoe_lock = lambda u: False
    bot._issue_snipe_with_lock = (
        lambda u, t: CaptureTheFlagBot._issue_snipe_with_lock(bot, u, t)
    )
    bot._issue_attack_on_unit = lambda u, t: CaptureTheFlagBot._issue_attack_on_unit(
        bot, u, t
    )
    bot._issue_heal_on_unit = lambda u, t: CaptureTheFlagBot._issue_heal_on_unit(
        bot, u, t
    )
    bot._issue_load_on_unit = lambda u, t: CaptureTheFlagBot._issue_load_on_unit(
        bot, u, t
    )
    bot._record_heal_cast = lambda u, t: CaptureTheFlagBot._record_heal_cast(bot, u, t)
    bot._mark_heal_command_attempt = (
        lambda u: CaptureTheFlagBot._mark_heal_command_attempt(bot, u)
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
    bot.last_action_was_attack_by_unit = {}
    bot._heal_lock_until_loop = lambda h, p: CaptureTheFlagBot._heal_lock_until_loop(
        bot, h, p
    )
    bot._game_loop = lambda: CaptureTheFlagBot._game_loop(bot)
    bot._HEAL_ABILITY_BY_TYPE = CaptureTheFlagBot._HEAL_ABILITY_BY_TYPE
    bot._TRANSPORT_LOAD_ABILITY = CaptureTheFlagBot._TRANSPORT_LOAD_ABILITY

    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (0.0, 0.0)
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot, agent


def _apply(bot, *, selected, command_idx, target_unit_tag, spatial_idx=100):
    return asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=list(selected),
            anchor_tag=selected[0],
            command_idx=command_idx,
            spatial_idx=spatial_idx,
            perspective_player_id=1,
            target_unit_tag=target_unit_tag,
        )
    )


def test_unit_target_heads_are_registered():
    assert "attack_unit" in UNIT_TARGET_COMMAND_HEADS
    assert "heal_unit" in UNIT_TARGET_COMMAND_HEADS
    assert "load_unit" in UNIT_TARGET_COMMAND_HEADS
    assert "snipe" in UNIT_TARGET_COMMAND_HEADS


def test_attack_unit_focus_fires_the_selected_tag():
    marine = _make_unit(1, UnitTypeId.MARINE, 50.0, 50.0, owner_id=1)
    far_ling = _make_unit(2, UnitTypeId.ZERGLING, 99.0, 99.0, owner_id=2)
    near_ling = _make_unit(3, UnitTypeId.ZERGLING, 50.5, 50.0, owner_id=2)
    bot, _ = _bot(marine, far_ling, near_ling)

    # Policy chose the *far* zergling. Agent-faithful actuation must honour it
    # instead of retargeting to the nearest enemy.
    _apply(bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=2)

    marine.attack.assert_called_once_with(far_ling)
    bot.do.assert_called_once()
    assert 1 in bot._command_attempt_tags["attack"]
    assert bot.last_action_was_move_by_unit[1] is False


def test_attack_unit_swarmhost_spawns_locusts_at_target_position():
    host = _make_unit(1, UnitTypeId.SWARMHOSTMP, 40.0, 40.0, owner_id=1)
    enemy = _make_unit(2, UnitTypeId.ZEALOT, 60.0, 70.0, owner_id=2)
    cast_cmd = object()
    host.return_value = cast_cmd
    bot, _ = _bot(host, enemy)

    _apply(bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=2)

    host.assert_called_once_with(AbilityId.EFFECT_SPAWNLOCUSTS, Point2((60.0, 70.0)))
    bot.do.assert_called_once_with(cast_cmd)
    host.attack.assert_not_called()


def test_attack_unit_never_spatial_decodes():
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=1)
    ling = _make_unit(2, UnitTypeId.ZERGLING, owner_id=2)
    bot, agent = _bot(marine, ling)

    _apply(bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=2)

    agent.convert_spatial_action_to_world.assert_not_called()


def test_attack_unit_skips_same_owner_target():
    """Under debug_control_enemy both armies look allied; ownership decides hostility."""
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=2)
    ally = _make_unit(2, UnitTypeId.MARAUDER, owner_id=2)
    bot, _ = _bot(marine, ally)

    ordered = _apply(bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=2)

    marine.attack.assert_not_called()
    bot.do.assert_not_called()
    assert ordered == []
    assert bot._unit_last_order == {}


def test_attack_unit_with_dead_target_is_a_no_op():
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=1)
    bot, _ = _bot(marine)

    _apply(bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=999)

    marine.attack.assert_not_called()
    bot.do.assert_not_called()


def test_heal_unit_transfuses_the_selected_ally_and_credits_cast():
    queen = _make_unit(1, UnitTypeId.QUEEN, owner_id=1, is_biological=True)
    wounded = _make_unit(
        2, UnitTypeId.ROACH, 80.0, 80.0, owner_id=1, health=30.0, is_biological=True
    )
    bot, _ = _bot(queen, wounded)

    # Range does not gate the order: the engine walks the Queen in.
    _apply(bot, selected=[1], command_idx=CMD_HEAL_UNIT, target_unit_tag=2)

    queen.assert_called_once_with(AbilityId.TRANSFUSION_TRANSFUSION, wounded)
    assert 1 in bot._heal_cast_tags
    assert 1 in bot._heal_order_tags
    assert 1 in bot._heal_attempt_tags
    assert bot._heal_lock[1][0] == 2


def test_heal_unit_respects_per_caster_biology():
    """heal_target_mask only promises *some* healer matches; the caster is re-checked."""
    scv = _make_unit(1, UnitTypeId.SCV, owner_id=1, is_mechanical=True)
    wounded_marine = _make_unit(
        2, UnitTypeId.MARINE, owner_id=1, health=10.0, is_biological=True
    )
    bot, _ = _bot(scv, wounded_marine)

    ordered = _apply(bot, selected=[1], command_idx=CMD_HEAL_UNIT, target_unit_tag=2)

    scv.assert_not_called()
    bot.do.assert_not_called()
    # Still counted as an attempt, so TensorBoard shows the wasted decision.
    assert 1 in bot._heal_attempt_tags
    assert ordered == []
    assert bot._unit_last_order == {}


def test_heal_unit_skips_full_health_target():
    queen = _make_unit(1, UnitTypeId.QUEEN, owner_id=1, is_biological=True)
    healthy = _make_unit(
        2,
        UnitTypeId.ROACH,
        owner_id=1,
        health=100.0,
        health_max=100.0,
        is_biological=True,
    )
    bot, _ = _bot(queen, healthy)

    ordered = _apply(bot, selected=[1], command_idx=CMD_HEAL_UNIT, target_unit_tag=2)

    queen.assert_not_called()
    assert 1 not in bot._heal_cast_tags
    assert ordered == []
    assert bot._unit_last_order == {}


def test_heal_unit_skips_enemy_target():
    queen = _make_unit(1, UnitTypeId.QUEEN, owner_id=1, is_biological=True)
    enemy = _make_unit(
        2, UnitTypeId.ROACH, owner_id=2, health=10.0, is_biological=True
    )
    bot, _ = _bot(queen, enemy)

    _apply(bot, selected=[1], command_idx=CMD_HEAL_UNIT, target_unit_tag=2)

    queen.assert_not_called()


def test_load_unit_transport_loads_the_selected_passenger():
    medivac = _make_unit(
        1, UnitTypeId.MEDIVAC, owner_id=1, cargo_max=8, cargo_used=0, cargo_size=0
    )
    far_marine = _make_unit(
        2, UnitTypeId.MARINE, 99.0, 99.0, owner_id=1, cargo_size=1
    )
    near_marine = _make_unit(
        3, UnitTypeId.MARINE, 50.5, 50.0, owner_id=1, cargo_size=1
    )
    bot, agent = _bot(medivac, far_marine, near_marine)

    _apply(bot, selected=[1], command_idx=CMD_LOAD_UNIT, target_unit_tag=2)

    medivac.assert_called_once_with(AbilityId.LOAD_MEDIVAC, far_marine)
    bot.do.assert_called_once()
    agent.convert_spatial_action_to_world.assert_not_called()
    assert 1 in bot._command_attempt_tags["load"]


def test_load_unit_passenger_smarts_onto_selected_transport():
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=1, cargo_size=1)
    medivac = _make_unit(
        2, UnitTypeId.MEDIVAC, owner_id=1, cargo_max=8, cargo_used=0, cargo_size=0
    )
    bot, _ = _bot(marine, medivac)

    _apply(bot, selected=[1], command_idx=CMD_LOAD_UNIT, target_unit_tag=2)

    marine.smart.assert_called_once_with(medivac)
    bot.do.assert_called_once()


def test_load_unit_skips_incompatible_pair():
    medivac = _make_unit(
        1, UnitTypeId.MEDIVAC, owner_id=1, cargo_max=8, cargo_used=8, cargo_size=0
    )
    marine = _make_unit(2, UnitTypeId.MARINE, owner_id=1, cargo_size=1)
    bot, _ = _bot(medivac, marine)

    _apply(bot, selected=[1], command_idx=CMD_LOAD_UNIT, target_unit_tag=2)

    medivac.assert_not_called()
    bot.do.assert_not_called()
    assert 1 in bot._command_attempt_tags["load"]


def test_unit_target_command_never_orders_a_unit_onto_itself():
    queen = _make_unit(1, UnitTypeId.QUEEN, owner_id=1, health=10.0, is_biological=True)
    other = _make_unit(2, UnitTypeId.QUEEN, owner_id=1, health=10.0, is_biological=True)
    bot, _ = _bot(queen, other)

    _apply(bot, selected=[1, 2], command_idx=CMD_HEAL_UNIT, target_unit_tag=1)

    queen.assert_not_called()
    other.assert_called_once_with(AbilityId.TRANSFUSION_TRANSFUSION, queen)


def test_unit_target_command_with_null_target_issues_nothing():
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=1)
    ling = _make_unit(2, UnitTypeId.ZERGLING, owner_id=2)
    bot, _ = _bot(marine, ling)

    ordered = _apply(
        bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=None
    )

    marine.attack.assert_not_called()
    bot.do.assert_not_called()
    # Must not claim the pointer selection as "ordered" — that blocks last-order
    # reissue and freezes opponents under opponent_decision_interval holds.
    assert ordered == []
    assert bot._unit_last_order == {}


def test_unit_target_command_with_missing_target_issues_nothing():
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=1)
    bot, _ = _bot(marine)

    ordered = _apply(
        bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=99
    )

    marine.attack.assert_not_called()
    bot.do.assert_not_called()
    assert ordered == []
    assert bot._unit_last_order == {}


def test_attack_unit_returns_ordered_tags_on_success():
    marine = _make_unit(1, UnitTypeId.MARINE, owner_id=1)
    ling = _make_unit(2, UnitTypeId.ZERGLING, owner_id=2)
    bot, _ = _bot(marine, ling)

    ordered = _apply(
        bot, selected=[1], command_idx=CMD_ATTACK_UNIT, target_unit_tag=2
    )

    assert ordered == [1]
    assert 1 in bot._unit_last_order


def test_shield_battery_is_never_ordered():
    battery = _make_unit(1, UnitTypeId.SHIELDBATTERY, owner_id=1, is_structure=True)
    wounded = _make_unit(2, UnitTypeId.ZEALOT, owner_id=1, health=10.0)
    bot, _ = _bot(battery, wounded)

    ordered = _apply(bot, selected=[1], command_idx=CMD_HEAL_UNIT, target_unit_tag=2)

    battery.assert_not_called()
    bot.do.assert_not_called()
    assert ordered == []