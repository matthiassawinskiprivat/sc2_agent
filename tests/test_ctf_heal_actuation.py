"""CTF heal_local actuation: Medivac/SCV/Queen abilities; battery/non-healers no-op."""
import asyncio
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.utils.unit_feature_helpers import HEAL_CELL_HALF


def _make_unit(
    tag,
    type_id,
    x,
    y,
    owner_id=1,
    health=50.0,
    health_max=100.0,
    radius=0.375,
    is_biological=None,
    is_mechanical=None,
):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.health = health
    u.health_max = health_max
    u.radius = radius
    u.is_structure = type_id == UnitTypeId.SHIELDBATTERY
    u.is_idle = False
    # Left unset (auto-mock truthy) unless a test cares about eligibility —
    # most tests here exercise click geometry / cast plumbing, not biology.
    if is_biological is not None:
        u.is_biological = is_biological
    if is_mechanical is not None:
        u.is_mechanical = is_mechanical
    cast_cmd = object()
    u.return_value = cast_cmd
    return u


def _bot_with_healers(healer, *allies, click=None):
    units = [healer, *allies]
    bot = MagicMock(spec=CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {
        "agent_architecture": "universal",
        "action_heads": ["move_global", "move_local", "attack_local", "heal_local"],
    }
    bot.actions = []
    bot.do = MagicMock()
    bot._unit_last_order = {}
    bot.last_action_was_move_by_unit = {}
    bot._heal_cast_tags = set()
    bot._heal_order_tags = set()
    bot._heal_attempt_tags = set()
    bot._heal_lock = {}
    bot.game_info = None
    bot.state = MagicMock()
    bot.state.game_loop = 0
    bot.frame_count = 0

    by_tag = {u.tag: u for u in units}

    def find_by_tag(tag):
        return by_tag.get(tag)

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = find_by_tag
    bot.units = list(units)
    bot.enemy_units = []

    # Bind real helpers onto the mock so apply_squad_action works.
    bot._issue_heal_at_point = lambda unit, target: CaptureTheFlagBot._issue_heal_at_point(
        bot, unit, target
    )
    bot._record_heal_cast = (
        lambda unit, patient: CaptureTheFlagBot._record_heal_cast(bot, unit, patient)
    )
    bot._mark_heal_command_attempt = (
        lambda unit: CaptureTheFlagBot._mark_heal_command_attempt(bot, unit)
    )
    bot._resolve_heal_target_in_cell = (
        lambda unit, target: CaptureTheFlagBot._resolve_heal_target_in_cell(
            bot, unit, target
        )
    )
    bot._heal_target_is_eligible = CaptureTheFlagBot._heal_target_is_eligible
    # spec= restricts attribute names but does not copy real class values, so
    # this stays a MagicMock (not the real ability table) unless rebound.
    bot._HEAL_ABILITY_BY_TYPE = CaptureTheFlagBot._HEAL_ABILITY_BY_TYPE
    bot._allied_units_for_owner = (
        lambda owner_id: CaptureTheFlagBot._allied_units_for_owner(bot, owner_id)
    )
    bot._unit_in_spatial_cell = CaptureTheFlagBot._unit_in_spatial_cell
    bot._refresh_heal_lock = lambda unit: CaptureTheFlagBot._refresh_heal_lock(bot, unit)
    bot._heal_lock_until_loop = (
        lambda healer, patient: CaptureTheFlagBot._heal_lock_until_loop(
            bot, healer, patient
        )
    )
    bot._game_loop = lambda: CaptureTheFlagBot._game_loop(bot)

    click_pos = click if click is not None else allies[0].position
    agent = MagicMock()
    agent.last_game_info = None
    agent.convert_spatial_action_to_world.return_value = (
        float(click_pos.x),
        float(click_pos.y),
    )
    bot.agent_system = MagicMock(unit_agents={"active_agent": agent})
    return bot


def _apply_squad_heal(bot, healer_tag):
    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[healer_tag],
            anchor_tag=healer_tag,
            command_idx=3,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )


def test_squad_heal_medivac_issues_heal():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, ally)
    bot.do.assert_called_once()
    healer.move.assert_not_called()
    assert 1 in bot._heal_cast_tags
    assert bot.last_action_was_move_by_unit[1] is False


def test_squad_heal_scv_issues_repair():
    healer = _make_unit(1, UnitTypeId.SCV, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 50.4, 50.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.EFFECT_REPAIR_SCV, ally)
    bot.do.assert_called_once()


def test_squad_heal_queen_issues_transfusion():
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.ROACH, 55.0, 50.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.TRANSFUSION_TRANSFUSION, ally)
    bot.do.assert_called_once()
    assert 1 in bot._heal_cast_tags
    assert 1 in bot._heal_attempt_tags


def test_squad_heal_non_healer_noop():
    marine = _make_unit(1, UnitTypeId.MARINE, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 52.0, 50.0)
    bot = _bot_with_healers(marine, ally)

    _apply_squad_heal(bot, 1)

    marine.assert_not_called()
    marine.move.assert_not_called()
    bot.do.assert_not_called()


def test_squad_heal_battery_noop():
    battery = _make_unit(1, UnitTypeId.SHIELDBATTERY, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.ZEALOT, 52.0, 50.0)
    bot = _bot_with_healers(battery, ally)

    _apply_squad_heal(bot, 1)

    battery.assert_not_called()
    battery.move.assert_not_called()
    bot.do.assert_not_called()


def test_squad_heal_no_ally_in_cell_is_noop():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 80.0, 80.0)
    click = Point2((60.0, 70.0))
    bot = _bot_with_healers(healer, ally, click=click)

    _apply_squad_heal(bot, 1)

    healer.assert_not_called()
    healer.move.assert_not_called()
    bot.do.assert_not_called()
    assert bot.last_action_was_move_by_unit[1] is False
    assert bot._heal_cast_tags == set()
    assert 1 in bot._heal_attempt_tags


def test_squad_heal_prefers_wounded_ally_in_cell():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    full = _make_unit(2, UnitTypeId.MARINE, 51.0, 50.0, health=100.0, health_max=100.0)
    hurt = _make_unit(3, UnitTypeId.MARINE, 51.2, 50.0, health=20.0, health_max=100.0)
    bot = _bot_with_healers(healer, full, hurt, click=Point2((51.1, 50.0)))

    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, hurt)


def test_squad_heal_medivac_cannot_heal_mechanical_ally():
    """Heal only affects Biological units in real SC2; a wounded Mechanical
    ally under the click must not resolve as a target — the order would
    silently do nothing in-game even though our code would otherwise credit
    heal_cast_reward and set the walk-in lock for a "cast" that never heals."""
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    thor = _make_unit(
        2, UnitTypeId.THOR, 52.0, 50.0, is_biological=False, is_mechanical=True
    )
    bot = _bot_with_healers(healer, thor)

    _apply_squad_heal(bot, 1)

    healer.assert_not_called()
    bot.do.assert_not_called()
    assert 1 not in bot._heal_cast_tags


def test_squad_heal_scv_cannot_repair_biological_ally():
    """Repair only affects Mechanical units/structures; a wounded Biological
    ally under the click must not resolve as a target for an SCV."""
    healer = _make_unit(1, UnitTypeId.SCV, 50.0, 50.0)
    marine = _make_unit(
        2, UnitTypeId.MARINE, 50.4, 50.0, is_biological=True, is_mechanical=False
    )
    bot = _bot_with_healers(healer, marine)

    _apply_squad_heal(bot, 1)

    healer.assert_not_called()
    bot.do.assert_not_called()
    assert 1 not in bot._heal_cast_tags


def test_squad_heal_medivac_skips_ineligible_ally_for_eligible_one_in_same_cell():
    """Eligibility filtering happens before the "most wounded" tie-break: a
    badly wounded Mechanical ally must not shadow a lightly wounded
    Biological ally that the Medivac can actually heal."""
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    thor = _make_unit(
        2,
        UnitTypeId.THOR,
        50.2,
        50.0,
        health=50.0,
        health_max=400.0,
        is_biological=False,
        is_mechanical=True,
    )
    marine = _make_unit(
        3,
        UnitTypeId.MARINE,
        50.3,
        50.0,
        health=90.0,
        health_max=100.0,
        is_biological=True,
        is_mechanical=False,
    )
    bot = _bot_with_healers(healer, thor, marine)

    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, marine)


def test_squad_heal_full_hp_in_cell_is_noop():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 52.0, 50.0, health=100.0, health_max=100.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)

    healer.assert_not_called()
    healer.move.assert_not_called()
    bot.do.assert_not_called()
    assert bot.last_action_was_move_by_unit[1] is False


def test_squad_heal_out_of_cast_range_still_issues_ability():
    """Clicking a wounded ally out of Heal range 4 still issues Heal; engine walks in."""
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 55.0, 50.0)  # dist 5 > 4
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, ally)
    healer.move.assert_not_called()
    assert 1 in bot._heal_cast_tags


def test_squad_heal_busy_skips_attack():
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)
    _apply_squad_heal(bot, 1)
    assert bot._heal_lock[1][0] == 2
    do_calls = bot.do.call_count

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=2,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    assert bot.do.call_count == do_calls
    assert bot._unit_last_order[1][0] == "heal_local"
    assert bot._heal_lock[1][0] == 2


def test_squad_heal_lock_clears_when_idle():
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)
    _apply_squad_heal(bot, 1)
    healer.is_idle = True

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=2,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    assert 1 not in bot._heal_lock


def test_squad_heal_busy_still_allows_new_heal():
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)
    _apply_squad_heal(bot, 1)
    healer.reset_mock()
    bot.do.reset_mock()
    _apply_squad_heal(bot, 1)

    healer.assert_called_once_with(AbilityId.TRANSFUSION_TRANSFUSION, ally)
    assert bot._heal_lock[1][0] == 2


def test_squad_heal_reissue_on_same_patient_does_not_recredit_cast():
    """Holding heal_local on a still-wounded ally must not pay heal_cast_reward
    every squad step; only a fresh engagement (new patient or lapsed lock) is
    reward-eligible. Otherwise camping heal out-earns combat by just parking."""
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)
    assert 1 in bot._heal_cast_tags
    bot._heal_cast_tags.clear()  # simulate the reward step consuming + clearing it

    _apply_squad_heal(bot, 1)  # still locked onto the same patient, loop unchanged
    assert 1 not in bot._heal_cast_tags
    assert bot._heal_lock[1][0] == 2


def test_squad_heal_reissue_on_same_patient_still_populates_order_tags():
    """heal_cast_reward must not re-fire while camping the same patient (above),
    but heal_reward's per-HP shaping must still see this step as "actively
    healing" — a Medivac beam / SCV repair only ticks HP on *continuing*
    steps, not the one that started the walk-in/lock, so gating heal_reward
    on the same fresh-cast-only set as heal_cast_reward would zero out nearly
    all HP credit for a multi-step channel."""
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)
    assert 1 in bot._heal_order_tags
    bot._heal_cast_tags.clear()
    bot._heal_order_tags.clear()  # simulate the reward step consuming + clearing it

    _apply_squad_heal(bot, 1)  # still locked onto the same patient, loop unchanged
    assert 1 not in bot._heal_cast_tags
    assert 1 in bot._heal_order_tags


def test_squad_heal_new_patient_recredits_cast_even_while_prior_lock_active():
    """Switching to a different wounded ally is a genuinely new cast."""
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally_a = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    ally_b = _make_unit(3, UnitTypeId.QUEEN, 52.0, 52.0)
    bot = _bot_with_healers(healer, ally_a, ally_b, click=ally_a.position)

    _apply_squad_heal(bot, 1)
    assert 1 in bot._heal_cast_tags
    bot._heal_cast_tags.clear()

    agent = bot.agent_system.unit_agents["active_agent"]
    agent.convert_spatial_action_to_world.return_value = (
        float(ally_b.position.x),
        float(ally_b.position.y),
    )
    _apply_squad_heal(bot, 1)
    assert 1 in bot._heal_cast_tags
    assert bot._heal_lock[1][0] == 3


def test_squad_heal_recredits_cast_after_lock_lapses():
    """Once the walk-in/animation window elapses, resuming on the same patient
    (e.g. re-wounded by ongoing combat) is a new cast again."""
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)

    _apply_squad_heal(bot, 1)
    until = bot._heal_lock[1][1]
    bot._heal_cast_tags.clear()

    bot.state.game_loop = until  # lock has lapsed
    _apply_squad_heal(bot, 1)
    assert 1 in bot._heal_cast_tags


def test_squad_heal_lock_expires_after_cast_window():
    """After the 1s Transfuse window, attack is allowed even if the Queen is busy."""
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    healer.movement_speed = 1.31
    ally = _make_unit(2, UnitTypeId.QUEEN, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)
    _apply_squad_heal(bot, 1)
    until = bot._heal_lock[1][1]
    assert until == CaptureTheFlagBot._heal_lock_until_loop(bot, healer, ally)
    bot.state.game_loop = until
    do_calls = bot.do.call_count

    asyncio.run(
        CaptureTheFlagBot.apply_squad_action(
            bot,
            selected_tags=[1],
            anchor_tag=1,
            command_idx=2,
            spatial_idx=100,
            perspective_player_id=1,
        )
    )

    assert 1 not in bot._heal_lock
    assert bot.do.call_count > do_calls
    assert bot._unit_last_order[1][0] == "attack_local"


def test_heal_lock_includes_walk_into_cast_range():
    healer = _make_unit(1, UnitTypeId.QUEEN, 50.0, 50.0)
    healer.movement_speed = 1.31
    ally = _make_unit(2, UnitTypeId.QUEEN, 60.0, 50.0)
    bot = _bot_with_healers(healer, ally)
    _apply_squad_heal(bot, 1)

    expected = CaptureTheFlagBot._heal_lock_until_loop(bot, healer, ally)
    assert bot._heal_lock[1][1] == expected
    anim_loops = (
        CaptureTheFlagBot._HEAL_CAST_ANIMATION_SECONDS[UnitTypeId.QUEEN]
        * CaptureTheFlagBot._HEAL_GAME_LOOPS_PER_SECOND
    )
    assert expected > anim_loops


def test_heal_cell_accepts_center_inside_one_cell():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 58.0, 70.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 60.4, 70.0)
    click = Point2((60.0, 70.0))
    bot = _bot_with_healers(healer, ally, click=click)

    target = CaptureTheFlagBot._resolve_heal_target_in_cell(bot, healer, click)
    assert target is ally
    assert HEAL_CELL_HALF == 0.5


def test_heal_cell_rejects_center_outside_one_cell():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 58.0, 70.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 60.6, 70.0, radius=0.375)
    click = Point2((60.0, 70.0))
    bot = _bot_with_healers(healer, ally, click=click)

    target = CaptureTheFlagBot._resolve_heal_target_in_cell(bot, healer, click)
    assert target is None


def test_heal_click_cells_three_hits_outside_one_cell():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 58.0, 70.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 61.2, 70.0, radius=0.375)
    click = Point2((60.0, 70.0))
    bot = _bot_with_healers(healer, ally, click=click)
    bot.training_conf["heal_click_cells"] = 3

    target = CaptureTheFlagBot._resolve_heal_target_in_cell(bot, healer, click)
    assert target is ally


def test_heal_click_hits_thor_body_outside_center_cell():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 60.0, 70.0)
    thor = _make_unit(2, UnitTypeId.THOR, 62.0, 70.0, radius=1.875)
    click = Point2((60.5, 70.0))  # 1.5 from Thor center; in body, not in 1×1 cell
    bot = _bot_with_healers(healer, thor, click=click)

    target = CaptureTheFlagBot._resolve_heal_target_in_cell(bot, healer, click)
    assert target is thor


def test_heal_click_misses_cyclone_outside_body_and_cell():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 60.0, 70.0)
    cyclone = _make_unit(2, UnitTypeId.CYCLONE, 62.0, 70.0, radius=0.75)
    click = Point2((60.5, 70.0))  # 1.5 from Cyclone center
    bot = _bot_with_healers(healer, cyclone, click=click)

    target = CaptureTheFlagBot._resolve_heal_target_in_cell(bot, healer, click)
    assert target is None


def test_apply_action_heal_medivac():
    healer = _make_unit(1, UnitTypeId.MEDIVAC, 50.0, 50.0)
    ally = _make_unit(2, UnitTypeId.MARINE, 52.0, 50.0)
    bot = _bot_with_healers(healer, ally)

    asyncio.run(CaptureTheFlagBot.apply_action(bot, 1, np.array([3, 100], dtype=np.int64)))

    healer.assert_called_once_with(AbilityId.MEDIVACHEAL_HEAL, ally)
    bot.do.assert_called_once()
