"""Level7 kiting shaping: full per-step attribution and kiting-simple shaping."""

import pytest
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.level_7_bot import Level7Bot


def _bare_bot(training_conf: dict) -> Level7Bot:
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.previous_health = {}
    bot.previous_allied_health = {}
    bot.previous_danger_counts = {}
    bot.last_attack_target_tag_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot.last_action_was_move_by_unit = {}
    bot._micro_under_fire_min_dist_prev = {}
    bot._allied_took_damage_this_step = {}
    bot._kiting_simple_prev_dist = {}
    bot._prev_enemy_in_attack_range = {}
    bot._prev_kiting_enemy_tag = {}
    bot._attack_order_weapon_cd_before = {}
    bot._pending_rewards = {}
    bot._enemy_last_known_pos = {}
    bot.current_assignments = {101: 201}
    return bot


def test_all_enemy_damage_attributed_to_actor():
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})
    bot.last_action_was_attack_by_unit[101] = False

    e1 = MagicMock(tag=201, health=8, shield=0, health_max=10, shield_max=0)
    e2 = MagicMock(tag=202, health=9, shield=0, health_max=10, shield_max=0)
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.previous_health = {201: 10.0, 202: 10.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(3.0)
    assert pooled == pytest.approx(0.0)


def test_retreat_shaping_no_fire_when_enemy_was_not_in_range():
    """Retreat reward does not fire when the enemy was NOT in attack range at step start."""
    training_conf = {"rewards": {"retreat_shaping_scale": 0.5}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((9, 10))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2((14, 10))
    enemy.ground_range = 1.0
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()

    # Pre-action snapshot: enemy NOT in attack range at step start.
    bot._kiting_simple_prev_dist[101] = 4.0
    bot._prev_enemy_in_attack_range[101] = False
    bot._prev_kiting_enemy_tag[101] = 201  # matches current assignment so guard doesn't clear state
    bot.last_action_was_move_by_unit[101] = True

    r = bot.calculate_kiting_simple_shaping(101)
    assert r == pytest.approx(0.0)


def test_retreat_shaping_fires_when_enemy_was_in_attack_range():
    """Retreat reward fires when enemy WAS in attack range at step start and unit moved away while still in own weapon range."""
    training_conf = {"rewards": {"retreat_shaping_scale": 0.5}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((9, 10))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2((14, 10))  # dist=5 ≤ ally_range=6 → still in weapon range
    enemy.ground_range = 1.0
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()

    # Pre-action snapshot: enemy WAS in attack range at step start (dist=4.0).
    bot._kiting_simple_prev_dist[101] = 4.0
    bot._prev_enemy_in_attack_range[101] = True
    bot._prev_kiting_enemy_tag[101] = 201  # matches current assignment
    bot.last_action_was_move_by_unit[101] = True

    # current dist = |9-14| = 5.0, prev_dist = 4.0, moved away by 1.0, still in range 6
    r = bot.calculate_kiting_simple_shaping(101)
    assert r == pytest.approx(0.5 * 1.0)


def test_retreat_shaping_no_fire_past_own_weapon_range():
    """Retreat reward must not fire when the unit has fled past its own weapon range (map-edge exploit)."""
    training_conf = {"rewards": {"retreat_shaping_scale": 0.5}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 10))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2((10, 10))  # dist=10 > ally_range=6 → out of weapon range
    enemy.ground_range = 5.0
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()

    # Pre-action snapshot: enemy was in attack range at step start (prev_dist=9, range=5).
    bot._kiting_simple_prev_dist[101] = 9.0
    bot._prev_enemy_in_attack_range[101] = True
    bot._prev_kiting_enemy_tag[101] = 201
    bot.last_action_was_move_by_unit[101] = True

    # Current dist=10 > prev_dist=9 (moved away), but dist=10 > ally_range=6 → no reward
    r = bot.calculate_kiting_simple_shaping(101)
    assert r == pytest.approx(0.0)


def test_sweet_spot_bonus_fires_in_sweet_spot():
    """Sweet-spot bonus fires when the unit is within own weapon range but outside enemy's attack range."""
    training_conf = {"rewards": {"sweet_spot_bonus": 0.1, "retreat_shaping_scale": 0.0}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 0))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    # dist=4: within ally range 6, but outside enemy threat radius max(1, 0.5*6)=3
    enemy = MagicMock(tag=201, position=Point2((4, 0)))
    enemy.ground_range = 1.0
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()

    r = bot.calculate_kiting_simple_shaping(101)
    # 1 enemy in sweet spot → 0.1 * 1 = 0.1
    assert r == pytest.approx(0.1)


def test_in_danger_penalty_fires_when_in_enemy_range():
    """In-danger penalty is subtracted for each enemy whose danger zone contains this unit."""
    training_conf = {"rewards": {"in_danger_penalty": 0.2, "retreat_shaping_scale": 0.0}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 0))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    # dist=1: well inside enemy threat radius max(4, 0.5*6)=4
    enemy = MagicMock(tag=201, position=Point2((1, 0)))
    enemy.ground_range = 4.0
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()

    r = bot.calculate_kiting_simple_shaping(101)
    # 1 enemy in danger zone → -(0.2 * 1) = -0.2
    assert r == pytest.approx(-0.2)


def test_close_in_penalty_is_proportional():
    """Close-in penalty scales with how much closer to the shooting enemy the agent moved."""
    training_conf = {"rewards": {"kiting_close_in_penalty": 0.4}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 0))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    # Enemy can shoot ally (within enemy range).
    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2((3, 0))
    enemy.ground_range = 4.0  # enemy range 4 > dist 3 → can shoot
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()
    bot.last_action_was_move_by_unit[101] = True

    # Pre-action snapshot: shooting-enemy min-dist was 4.0 at step start.
    bot._micro_under_fire_min_dist_prev[101] = 4.0
    bot._prev_kiting_enemy_tag[101] = 201  # matches current assignment

    # Stub _min_dist_to_shooting_enemies_in_fovea to return the post-step distance.
    bot._min_dist_to_shooting_enemies_in_fovea = MagicMock(return_value=3.0)

    r = bot.calculate_kiting_simple_shaping(101)
    # penalty = 0.4 * (4.0 - 3.0) = 0.4
    assert r == pytest.approx(-0.4)


def test_weapon_ready_attack_bonus_fires_when_cooldown_zero():
    """weapon_ready_attack_bonus still fires when explicitly set to a non-zero value."""
    training_conf = {
        "rewards": {
            "weapon_ready_attack_bonus": 0.5,
            "retreat_shaping_scale": 0.0,
        }
    }
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 0))
    ally.ground_range = 6.0
    ally.air_range = 0.0
    ally.is_flying = False

    enemy = MagicMock(tag=201, position=Point2((5, 0)))
    enemy.ground_range = 1.0
    enemy.air_range = 0.0
    enemy.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.refresh_combat_assignments = MagicMock()

    bot.last_action_was_attack_by_unit[101] = True
    bot.last_action_was_move_by_unit[101] = False
    bot._attack_order_weapon_cd_before[101] = 0.0

    r = bot.calculate_kiting_simple_shaping(101)
    assert r == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Regression: melee enemy attack-range check must account for collision radii
# ---------------------------------------------------------------------------

def test_prev_enemy_in_attack_range_set_true_for_melee_contact():
    """
    record_pre_action_kiting_state must set _prev_enemy_in_attack_range=True when a
    melee enemy (ground_range=0.1, zergling-like) is within physical contact distance.

    Previously this used `dist <= enemy_range + eps` (0.1), which is smaller than the
    physical minimum center-to-center distance (~1.0), so the flag was always False and
    the retreat reward never fired for Hellion vs Zergling matchups.
    """
    training_conf = {"rewards": {}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((5.0, 0.0))
    ally.radius = 0.625          # Hellion collision radius
    ally.ground_range = 5.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    # Zergling at physical contact: dist ≈ 0.8, well within radius_sum (0.625+0.375=1.0)
    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2((5.8, 0.0))
    enemy.radius = 0.375         # Zergling collision radius
    enemy.ground_range = 0.1    # raw SC2 weapon data — much smaller than physical contact dist
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.current_assignments = {101: 201}
    bot.refresh_combat_assignments = MagicMock()

    # record_pre_action_kiting_state is the new setter for _prev_enemy_in_attack_range.
    bot.record_pre_action_kiting_state(101)

    # Effective range = max(0.1, 0.375+0.625) = 1.0; dist 0.8 ≤ 1.0 → must be True.
    assert bot._prev_enemy_in_attack_range.get(101) is True


def test_prev_enemy_in_attack_range_false_when_out_of_melee_contact():
    """
    record_pre_action_kiting_state must set _prev_enemy_in_attack_range=False when the
    melee enemy is far enough away that the effective contact radius is not reached.
    """
    training_conf = {"rewards": {}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0.0, 0.0))
    ally.radius = 0.625
    ally.ground_range = 5.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    # Zergling at dist=3.0 — well outside melee contact (~1.0)
    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2((3.0, 0.0))
    enemy.radius = 0.375
    enemy.ground_range = 0.1
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.current_assignments = {101: 201}
    bot.refresh_combat_assignments = MagicMock()

    bot.record_pre_action_kiting_state(101)

    # Effective range = max(0.1, 1.0) = 1.0; dist 3.0 > 1.0 → must be False.
    assert bot._prev_enemy_in_attack_range.get(101) is False


def test_retreat_fires_after_melee_contact_with_zergling():
    """
    Full retreat reward chain for a melee enemy: record pre-action state at contact
    distance (sets _prev_enemy_in_attack_range=True), then simulate the move step and
    verify the retreat reward fires against the pre-action baseline.
    """
    training_conf = {"rewards": {"retreat_shaping_scale": 0.5}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.radius = 0.625
    ally.ground_range = 5.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    enemy = MagicMock()
    enemy.tag = 201
    enemy.radius = 0.375
    enemy.ground_range = 0.1    # zergling raw weapon data
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.exists = True
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.find_by_tag = MagicMock(return_value=enemy)

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.enemy_units = MagicMock(return_value=mock_enemy_grp)
    bot.current_assignments = {101: 201}
    bot.refresh_combat_assignments = MagicMock()

    # --- Pre-action snapshot: zergling is in physical contact (dist 0.8 ≤ effective_range 1.0) ---
    ally.position = Point2((0.8, 0.0))
    enemy.position = Point2((0.0, 0.0))
    bot.record_pre_action_kiting_state(101)
    assert bot._prev_enemy_in_attack_range[101] is True  # contact confirmed

    # --- Simulate game step: ally moves away to dist=4.0 (still within own range 5) ---
    # Ally position is now the post-step position; pre-action snapshot holds dist=0.8.
    ally.position = Point2((4.0, 0.0))
    bot.last_action_was_move_by_unit[101] = True    # move step

    r = bot.calculate_kiting_simple_shaping(101)
    # prev_dist=0.8 (pre-action), new_dist=4.0 (post-step), moved away by 3.2, still in range (4.0 ≤ 5.0)
    assert r == pytest.approx(0.5 * (4.0 - 0.8))


# ---------------------------------------------------------------------------
# Regression: kiting state must be invalidated when the assignment changes
# ---------------------------------------------------------------------------

def test_kiting_state_cleared_on_assignment_change():
    """
    When the assigned enemy changes mid-step (e.g. old enemy dies during the game step),
    the pre-action snapshot for the OLD enemy must not produce a spurious retreat reward
    against the NEW enemy.

    The assignment-change guard in calculate_kiting_simple_shaping detects that
    _prev_kiting_enemy_tag (set by record_pre_action_kiting_state before the step) no
    longer matches the post-step e_tag, and clears the stale snapshot.
    """
    training_conf = {"rewards": {"retreat_shaping_scale": 0.5}}
    bot = _bare_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.radius = 0.625
    ally.ground_range = 5.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    enemy_old = MagicMock()
    enemy_old.tag = 201
    enemy_old.radius = 0.375
    enemy_old.ground_range = 0.1
    enemy_old.air_range = 0.0
    enemy_old.is_flying = False
    enemy_old.position = Point2((0.8, 0.0))

    enemy_new = MagicMock()
    enemy_new.tag = 999
    enemy_new.radius = 0.375
    enemy_new.ground_range = 0.1
    enemy_new.air_range = 0.0
    enemy_new.is_flying = False
    enemy_new.position = Point2((4.0, 0.0))

    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = ally
    bot.refresh_combat_assignments = MagicMock()

    # --- Pre-action snapshot: enemy_old at contact distance ---
    ally.position = Point2((0.8, 0.0))
    bot.current_assignments = {101: 201}
    mock_grp_old = MagicMock()
    mock_grp_old.exists = True
    mock_grp_old.__iter__.return_value = [enemy_old]
    mock_grp_old.find_by_tag = MagicMock(side_effect=lambda t: enemy_old if t == 201 else None)
    bot.enemy_units = MagicMock(return_value=mock_grp_old)
    bot.record_pre_action_kiting_state(101)
    assert bot._prev_kiting_enemy_tag[101] == 201  # snapshot recorded for enemy_old

    # --- Simulate game step: enemy_old dies, assignment switches to enemy_new ---
    bot.current_assignments = {101: 999}
    ally.position = Point2((4.0, 0.0))
    mock_grp_new = MagicMock()
    mock_grp_new.exists = True
    mock_grp_new.__iter__.return_value = [enemy_new]
    mock_grp_new.find_by_tag = MagicMock(side_effect=lambda t: enemy_new if t == 999 else None)
    bot.enemy_units = MagicMock(return_value=mock_grp_new)
    bot.last_action_was_move_by_unit[101] = True
    r = bot.calculate_kiting_simple_shaping(101)

    # Assignment changed (201 → 999): stale snapshot cleared → retreat reward must NOT fire.
    assert r == pytest.approx(0.0), (
        "Spurious retreat reward fired with stale state from old enemy after assignment change"
    )
