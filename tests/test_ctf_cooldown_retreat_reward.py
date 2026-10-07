"""
Regression tests for CTF cooldown-retreat shaping (tactical fovea enemy CoM).
"""
import math
import pytest
from unittest.mock import MagicMock
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(
    tag: int,
    x: float,
    y: float,
    owner_id: int = 1,
    weapon_cd: float = 0.0,
    ground_range: float = 5.0,
    type_id=None,
):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = owner_id
    u.is_structure = False
    u.health = 20.0
    u.health_max = 100.0
    u.shield = 0.0
    u.shield_max = 0.0
    u.weapon_cooldown = weapon_cd
    u.ground_range = ground_range
    u.air_range = 0.0
    u.is_flying = False
    if type_id is not None:
        u.type_id = type_id
    return u


def _make_bot(scale: float = 0.8) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "rewards": {
                "cooldown_retreat_scale": scale,
                "detection_reward": 0.0,
                "aoe_land_reward": 0.0,
                "defender_zone_leave_penalty": 0.0,
                "objective_credit_min_scale": 1.0,
                "distance_multiplier": 0.0,
            }
        },
    )
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)

    def _struct_col(pylon=None):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        return MagicMock(return_value=col)

    bot.structures = _struct_col()
    bot.enemy_structures = _struct_col()
    return bot


def _arm_cooldown_retreat(
    bot,
    tag: int,
    *,
    cd: float,
    prev_dist: float,
    com: Point2,
    moved: bool = True,
    attacked_prev: bool = False,
    prev_vit: float = 100.0,
) -> None:
    bot._pre_action_weapon_cd_by_unit[tag] = cd
    bot._pre_action_dist_to_enemy_com[tag] = prev_dist
    bot._pre_action_enemy_com[tag] = com
    bot._pre_action_attacked_prev_step[tag] = attacked_prev
    bot._pre_action_vitality[tag] = prev_vit
    bot.last_action_was_move_by_unit[tag] = moved
    bot.last_action_was_attack_by_unit[tag] = False


def test_cooldown_retreat_fires_when_moving_away_from_enemy_com():
    bot = _make_bot(scale=0.8)
    ally = _make_unit(101, 5.0, 0.0, owner_id=1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = ally
    bot.units = [ally]
    bot.enemy_units = [enemy]
    bot.active_unit_tag = 101

    _arm_cooldown_retreat(
        bot, 101, cd=7.0, prev_dist=4.0, com=Point2((0.0, 0.0))
    )

    reward = bot.calculate_cooldown_retreat_reward(101, [enemy])
    assert reward == pytest.approx(0.8 * 0.8 * (5.0 - 4.0))


def test_no_reward_for_melee_unit():
    bot = _make_bot(scale=0.8)
    ally = _make_unit(101, 6.0, 0.0, ground_range=0.1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2, ground_range=0.1)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0))
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_for_helliontank_even_if_ground_range_is_large():
    bot = _make_bot(scale=0.8)
    # Our unit_data gives HellionTank (Hellbat) a weapon range > melee threshold,
    # but it should still be treated as melee for cooldown-kite shaping.
    ally = _make_unit(
        101,
        6.0,
        0.0,
        owner_id=1,
        weapon_cd=5.0,
        ground_range=2.0,
        type_id=UnitTypeId.HELLIONTANK,
    )
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2, ground_range=2.0)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0))
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_for_archon_even_if_ground_range_is_large():
    bot = _make_bot(scale=0.8)
    # Archon catalog range is 3.0 (> melee threshold 1.5) but CTF treats it as melee.
    ally = _make_unit(
        101,
        6.0,
        0.0,
        owner_id=1,
        weapon_cd=5.0,
        ground_range=3.0,
        type_id=UnitTypeId.ARCHON,
    )
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2, ground_range=3.0)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0))
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0
    assert bot._is_melee_combat_unit(ally, [enemy]) is True


def test_no_reward_without_vitality_loss():
    bot = _make_bot()
    ally = _make_unit(101, 6.0, 0.0)
    ally.health = 100.0
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0)), prev_vit=100.0
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_when_attack_command_issued():
    bot = _make_bot()
    ally = _make_unit(101, 6.0, 0.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0)), moved=False
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_when_moving_toward_enemy_com():
    bot = _make_bot()
    ally = _make_unit(101, 3.0, 0.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=5.0, prev_dist=5.0, com=Point2((0.0, 0.0))
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_without_frozen_enemy_com_snapshot():
    """Pre-action fovea miss leaves no frozen CoM — no retreat credit."""
    bot = _make_bot()
    ally = _make_unit(101, 50.0, 50.0)
    enemy = _make_unit(201, 70.0, 50.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    bot._pre_action_weapon_cd_by_unit[101] = 5.0
    bot._pre_action_dist_to_enemy_com[101] = 20.0
    # Intentionally omit _pre_action_enemy_com (enemies were outside fovea).
    bot.last_action_was_move_by_unit[101] = True

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_retreat_rewarded_when_faster_enemies_chase_and_close_live_gap():
    """Marine walks away from pre-action CoM; Zerglings chase and shrink live gap.

    Old live-CoM gate would see curr_live_dist < prev_dist and pay 0. Frozen CoM
    still credits the unit's own retreat — the signal we want for kite learning.
    """
    bot = _make_bot(scale=0.9)
    # Marine retreated from x=5 to x=7 (away from CoM at origin).
    ally = _make_unit(101, 7.0, 0.0, owner_id=1)
    # Zergling chased from x=0 to x=4 — live gap is now 3 < prev_dist 5.
    enemy = _make_unit(201, 4.0, 0.0, owner_id=2, ground_range=0.1)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot, 101, cd=7.0, prev_dist=5.0, com=Point2((0.0, 0.0))
    )

    live_gap = bot._dist_to_point(ally, enemy.position)
    assert live_gap < 5.0  # chase closed the live gap

    reward = bot.calculate_cooldown_retreat_reward(101, [enemy])
    assert reward == pytest.approx(0.9 * 0.8 * (7.0 - 5.0))


def test_retreat_rewarded_when_under_fire_even_if_weapon_ready():
    """Vitality loss is the under-fire gate, not leftover weapon cooldown."""
    bot = _make_bot(scale=0.8)
    ally = _make_unit(101, 6.0, 0.0, owner_id=1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    _arm_cooldown_retreat(
        bot,
        101,
        cd=0.0,
        prev_dist=4.0,
        com=Point2((0.0, 0.0)),
        attacked_prev=False,
    )

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == pytest.approx(
        0.8 * 0.8 * (6.0 - 4.0)
    )


def test_low_hp_retreat_pays_more_than_high_hp():
    bot = _make_bot(scale=1.0)
    low = _make_unit(101, 6.0, 0.0, owner_id=1)
    low.health = 20.0
    high = _make_unit(102, 6.0, 0.0, owner_id=1)
    high.health = 80.0
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)

    bot.all_units.find_by_tag = MagicMock(return_value=low)
    _arm_cooldown_retreat(bot, 101, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0)))
    low_reward = bot.calculate_cooldown_retreat_reward(101, [enemy])

    bot.all_units.find_by_tag = MagicMock(return_value=high)
    _arm_cooldown_retreat(bot, 102, cd=5.0, prev_dist=4.0, com=Point2((0.0, 0.0)))
    high_reward = bot.calculate_cooldown_retreat_reward(102, [enemy])

    assert low_reward == pytest.approx(0.8 * (6.0 - 4.0))
    assert high_reward == pytest.approx(0.2 * (6.0 - 4.0))
    assert low_reward > high_reward


def test_record_snapshots_previous_attack_flag():
    bot = _make_bot()
    ally = _make_unit(101, 8.0, 0.0, weapon_cd=0.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.enemy_units = [enemy]
    bot.units = [ally]
    bot.last_action_was_attack_by_unit[101] = True

    bot.record_pre_action_retreat_state(101)

    assert bot._pre_action_weapon_cd_by_unit[101] == 0.0
    assert bot._pre_action_attacked_prev_step[101] is True


def test_enemy_com_uses_all_fovea_enemies():
    bot = _make_bot(scale=1.0)
    ally = _make_unit(101, 10.0, 0.0)
    e1 = _make_unit(201, 0.0, 0.0, owner_id=2)
    e2 = _make_unit(202, 0.0, 4.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    com = Point2((0.0, 2.0))
    com_dist = math.hypot(10.0 - com.x, 0.0 - com.y)

    _arm_cooldown_retreat(
        bot, 101, cd=3.0, prev_dist=com_dist - 1.0, com=com
    )

    reward = bot.calculate_cooldown_retreat_reward(101, [e1, e2])
    assert reward == pytest.approx(0.8)


def test_record_pre_action_retreat_state_snapshots_baseline():
    bot = _make_bot()
    ally = _make_unit(101, 8.0, 0.0, weapon_cd=6.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.enemy_units = [enemy]
    bot.units = [ally]

    bot.record_pre_action_retreat_state(101)

    assert bot._pre_action_weapon_cd_by_unit[101] == 6.0
    assert bot._pre_action_dist_to_enemy_com[101] == pytest.approx(8.0)
    frozen = bot._pre_action_enemy_com[101]
    assert frozen.x == pytest.approx(0.0)
    assert frozen.y == pytest.approx(0.0)


def test_record_pre_action_clears_when_no_fovea_enemies():
    bot = _make_bot()
    ally = _make_unit(101, 50.0, 50.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.enemy_units = [enemy]
    bot.units = [ally]

    bot.record_pre_action_retreat_state(101)

    assert 101 not in bot._pre_action_dist_to_enemy_com
    assert 101 not in bot._pre_action_enemy_com


def test_integrated_in_calculate_reward():
    bot = _make_bot(scale=0.5)
    ally = _make_unit(101, 6.0, 0.0, owner_id=1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    own_pylon = _make_unit(888, 0.0, 0.0)
    own_pylon.is_structure = True
    own_pylon.health = 20.0
    own_pylon.shield = 0.0
    enemy_pylon = _make_unit(999, 40.0, 0.0, owner_id=2)
    enemy_pylon.is_structure = True
    enemy_pylon.health = 20.0
    enemy_pylon.shield = 0.0

    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.units = [ally]
    bot.enemy_units = [enemy]
    bot.active_unit_tag = 101
    bot.unit_command_map = {101: "attack"}
    bot.prev_dist_per_squad = {}

    def _struct_col(pylon):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        return MagicMock(return_value=col)

    bot.structures = _struct_col(own_pylon)
    bot.enemy_structures = _struct_col(enemy_pylon)

    _arm_cooldown_retreat(
        bot, 101, cd=4.0, prev_dist=5.0, com=Point2((0.0, 0.0))
    )

    reward = bot.calculate_reward(perspective_player_id=1)
    expected_retreat = 0.5 * 0.8 * (6.0 - 5.0)
    step_penalty = -0.05
    assert reward == pytest.approx(expected_retreat + step_penalty)


def test_cooldown_retreat_info_accumulates_episode_and_emits_on_done():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env._episode_cooldown_retreat = 0.0
    env.bot = _make_bot(scale=0.8)
    env.bot._step_cooldown_retreat_reward = 1.6

    step = env._record_cooldown_retreat_metrics(credit_trainee=True)
    assert step == pytest.approx(1.6)
    assert env._episode_cooldown_retreat == pytest.approx(1.6)
    assert env.bot._step_cooldown_retreat_reward == 0.0
    mid = env._heal_cast_info(0, done=False, truncated=False, step_cooldown_retreat=step)
    assert mid["cooldown_retreat"] == pytest.approx(1.6)
    assert "cooldown_retreat_episode" not in mid

    env.bot._step_cooldown_retreat_reward = 0.4
    # Opponent physics step: pop/clear but do not credit the trainee episode.
    skipped = env._record_cooldown_retreat_metrics(credit_trainee=False)
    assert skipped == 0.0
    assert env._episode_cooldown_retreat == pytest.approx(1.6)

    env.bot._step_cooldown_retreat_reward = 0.4
    step2 = env._record_cooldown_retreat_metrics(credit_trainee=True)
    terminal = env._heal_cast_info(
        0, done=True, truncated=False, step_cooldown_retreat=step2
    )
    assert terminal["cooldown_retreat"] == pytest.approx(0.4)
    assert terminal["cooldown_retreat_episode"] == pytest.approx(2.0)


def test_squad_step_reward_tracks_cooldown_retreat_for_metrics():
    bot = _make_bot(scale=0.8)
    ally = _make_unit(101, 5.0, 0.0, owner_id=1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = ally
    bot.units = [ally]
    bot.enemy_units = [enemy]
    bot.active_unit_tag = 101
    bot.unit_command_map = {101: "attack"}
    bot.prev_dist_per_squad = {}
    bot.prev_own_unit_count = {1: 1, 2: 0}
    bot.prev_enemy_unit_count = {1: 1, 2: 0}
    bot.prev_own_units_health = {1: 100.0, 2: 0.0}
    bot.prev_enemy_units_health = {1: 100.0, 2: 0.0}
    _arm_cooldown_retreat(
        bot, 101, cd=7.0, prev_dist=4.0, com=Point2((0.0, 0.0))
    )

    bot.calculate_squad_step_reward([101], perspective_player_id=1)
    assert bot.pop_cooldown_retreat_reward() == pytest.approx(0.8 * 0.8 * (5.0 - 4.0))
    assert bot.pop_cooldown_retreat_reward() == 0.0
