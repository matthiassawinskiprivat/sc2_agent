"""Attacker wins by clearing the enemy objective zone, not by destroying the pylon.

The flag pylon is the zone anchor (``attacker_objective_radius``). Combat mobiles
and non-flag buildings inside the zone block the clear; the pylon itself does not.
"""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

WIN = 15.0
DRAW = -10.0
LOSS = -12.5
RADIUS = 12.0

_REWARDS = {
    "win_reward": WIN,
    "loss_penalty": LOSS,
    "draw_penalty": DRAW,
    "attacker_objective_radius": RADIUS,
}


def _unit(
    tag,
    x,
    y,
    *,
    owner_id,
    type_id=UnitTypeId.ZERGLING,
    is_structure=False,
    is_flying=False,
):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = owner_id
    u.type_id = type_id
    u.is_structure = is_structure
    u.is_flying = is_flying
    u.health = 50.0
    return u


@pytest.fixture
def env():
    e = CaptureTheFlagSingleEnv(visualize=False, max_steps=100)
    e.bot = MagicMock()
    e.bot.structures = MagicMock()
    e.bot.enemy_structures = MagicMock()
    e.bot.unit_command_map = {}
    e.bot.enemy_pylon_pos = {1: None, 2: None}
    e.unit_pools_config = [
        {"agent": True, "high_level_command": "attack", "region": "A"},
        {"high_level_command": "defend", "region": "A"},
    ]
    e.label_to_defender_pos = {"A": Point2((50.0, 50.0))}
    e.player_units = {1: [10], 2: [20]}
    e.unit_pool_index = {10: 0, 20: 1}
    e.bot.unit_command_map = {10: "attack", 20: "defend"}
    return e


def _set_units(env, units):
    env.bot.all_units = MagicMock()
    env.bot.all_units.__iter__ = MagicMock(return_value=iter(units))
    # ``_has_combat_units`` uses ``.filter(...).amount``
    def _filter(pred):
        matched = [u for u in units if pred(u)]
        out = MagicMock()
        out.amount = len(matched)
        out.__iter__ = MagicMock(return_value=iter(matched))
        return out

    env.bot.all_units.filter = _filter


def test_enemy_army_wipe_with_pylon_standing_is_not_win(env):
    """Army wipe alone is not enough — the flag pylon in the zone still blocks."""
    pylon = _unit(99, 50, 50, owner_id=2, type_id=UnitTypeId.PYLON, is_structure=True)
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, pylon])
    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=False,
        p1_army_lost=False,
        p2_army_lost=True,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is False
    assert delta == 0.0


def test_enemy_pylon_destroy_alone_is_not_win(env):
    """Destroying the pylon while enemy units remain in the zone does not win."""
    ling = _unit(20, 50, 50, owner_id=2)  # still in the zone
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, ling])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=True,  # enemy flag gone
        p1_army_lost=False,
        p2_army_lost=False,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is False
    assert delta == 0.0


def test_clearing_all_units_and_buildings_from_zone_is_win(env):
    """Zone empty of enemy units and buildings (pylon gone) → attacker win."""
    reaper = _unit(10, 48, 48, owner_id=1, type_id=UnitTypeId.REAPER)
    # Enemy ling fled outside the radius; pylon already destroyed.
    ling = _unit(20, 80, 80, owner_id=2)
    _set_units(env, [reaper, ling])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=True,
        p1_army_lost=False,
        p2_army_lost=False,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == WIN


def test_flying_units_in_the_zone_do_not_block_the_win(env):
    """Ground units and the pylon must die. Air left in the zone does not."""
    reaper = _unit(10, 48, 48, owner_id=1, type_id=UnitTypeId.REAPER)
    mutalisk = _unit(
        20, 50, 50, owner_id=2, type_id=UnitTypeId.MUTALISK, is_flying=True
    )
    _set_units(env, [reaper, mutalisk])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True

    ling = _unit(21, 50, 50, owner_id=2, type_id=UnitTypeId.ZERGLING, is_flying=False)
    _set_units(env, [reaper, mutalisk, ling])
    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False

    pylon = _unit(99, 50, 50, owner_id=2, type_id=UnitTypeId.PYLON, is_structure=True)
    _set_units(env, [reaper, mutalisk, pylon])
    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


def test_air_unit_with_missing_flying_flag_does_not_block(env):
    """Liberator defender mode and fog snapshots leave ``is_flying`` false."""
    reaper = _unit(10, 48, 48, owner_id=1, type_id=UnitTypeId.REAPER)
    mutalisk = _unit(
        20, 50, 50, owner_id=2, type_id=UnitTypeId.MUTALISK, is_flying=False
    )
    liberator = _unit(
        21, 51, 50, owner_id=2, type_id=UnitTypeId.LIBERATORAG, is_flying=False
    )
    _set_units(env, [reaper, mutalisk, liberator])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True


def test_landed_viking_still_blocks_the_zone(env):
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    viking = _unit(
        20, 50, 50, owner_id=2, type_id=UnitTypeId.VIKINGASSAULT, is_flying=False
    )
    _set_units(env, [reaper, viking])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


def test_locust_and_broodling_do_not_hold_a_cleared_zone(env):
    """Summons from a host outside the zone must not veto the win."""
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    locust = _unit(21, 50, 50, owner_id=2, type_id=UnitTypeId.LOCUSTMP)
    broodling = _unit(22, 51, 50, owner_id=2, type_id=UnitTypeId.BROODLING)
    _set_units(env, [reaper, locust, broodling])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._has_combat_units(2) is False
    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=True,
        p1_army_lost=False,
        p2_army_lost=True,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == WIN


def test_snapshot_of_a_dead_ground_unit_does_not_block(env):
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    ghost = _unit(20, 50, 50, owner_id=2, type_id=UnitTypeId.ZERGLING)
    ghost.is_snapshot = True
    _set_units(env, [reaper, ghost])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True


def test_p2_trainee_wins_when_only_air_remains_in_the_enemy_zone(env):
    """Player-2 trainee still wins after the enemy ground army and pylon are gone."""
    env.unit_pools_config = [
        {"agent": True, "high_level_command": "defend", "region": "A"},
        {"high_level_command": "attack", "region": "A"},
    ]
    mutalisk = _unit(
        10, 50, 50, owner_id=1, type_id=UnitTypeId.MUTALISK, is_flying=False
    )
    reaper = _unit(20, 80, 80, owner_id=2, type_id=UnitTypeId.REAPER)
    _set_units(env, [mutalisk, reaper])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=True,
        p1_army_lost=False,
        p2_army_lost=False,
        perspective_player_id=2,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == WIN


def test_missing_anchor_with_empty_enemy_board_is_a_win(env):
    env.unit_pools_config = []
    env.label_to_defender_pos = {}
    env.bot.enemy_pylon_pos = {1: None, 2: None}
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper])

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=False,
        p1_army_lost=False,
        p2_army_lost=True,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == WIN


def test_missing_anchor_with_an_enemy_ling_is_not_a_win(env):
    env.unit_pools_config = []
    env.label_to_defender_pos = {}
    env.bot.enemy_pylon_pos = {1: None, 2: None}
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    ling = _unit(20, 10, 10, owner_id=2)
    _set_units(env, [reaper, ling])

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


def test_enemy_unit_inside_zone_blocks_win(env):
    pylon = _unit(99, 50, 50, owner_id=2, type_id=UnitTypeId.PYLON, is_structure=True)
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    ling = _unit(20, 52, 52, owner_id=2)  # inside radius
    _set_units(env, [reaper, ling, pylon])

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=False,
        p1_army_lost=False,
        p2_army_lost=False,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is False
    assert delta == 0.0


def test_flag_pylon_inside_zone_blocks_clear(env):
    """A standing flag pylon is a building and must be cleared like any other."""
    pylon = _unit(99, 50, 50, owner_id=2, type_id=UnitTypeId.PYLON, is_structure=True)
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, pylon])

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


@pytest.mark.parametrize(
    "type_id,is_structure",
    [
        (UnitTypeId.CREEPTUMORBURROWED, False),
        (UnitTypeId.CREEPTUMOR, True),
        (UnitTypeId.CREEPTUMORQUEEN, True),
    ],
)
def test_creep_tumor_inside_zone_does_not_block_win(env, type_id, is_structure):
    """Heal tumors spread creep; they are not part of the zone-clear win."""
    tumor = _unit(21, 52, 52, owner_id=2, type_id=type_id, is_structure=is_structure)
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, tumor])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is True
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=True,
        p1_army_lost=False,
        p2_army_lost=True,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == WIN


def test_creep_tumor_does_not_excuse_a_standing_pylon(env):
    tumor = _unit(
        21, 52, 52, owner_id=2, type_id=UnitTypeId.CREEPTUMORBURROWED, is_structure=False
    )
    pylon = _unit(99, 50, 50, owner_id=2, type_id=UnitTypeId.PYLON, is_structure=True)
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, tumor, pylon])

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


def test_free_cloak_detector_does_not_block_clear(env):
    """Observer / Raven / Overseer are sensors, not fighting units."""
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    for type_id in (UnitTypeId.OBSERVER, UnitTypeId.RAVEN, UnitTypeId.OVERSEER):
        sensor = _unit(30, 50, 50, owner_id=2, type_id=type_id)
        sensor.is_cloaked = True
        sensor.is_burrowed = False
        sensor.is_detected = False
        _set_units(env, [reaper, sensor])
        env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}
        assert env._enemy_objective_zone_cleared(1, _REWARDS) is True


def test_undetected_dark_templar_in_zone_blocks_clear(env):
    """A cloaked combat unit holds the zone whether or not it is revealed."""
    dt = _unit(31, 52, 50, owner_id=2, type_id=UnitTypeId.DARKTEMPLAR)
    dt.is_cloaked = True
    dt.is_burrowed = False
    dt.is_detected = False
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, dt])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


def test_non_flag_building_in_zone_blocks_clear(env):
    bunker = _unit(
        98, 51, 51, owner_id=2, type_id=UnitTypeId.BUNKER, is_structure=True
    )
    reaper = _unit(10, 40, 40, owner_id=1, type_id=UnitTypeId.REAPER)
    _set_units(env, [reaper, bunker])
    env.label_to_defender_pos = {"A": Point2((50.0, 50.0))}

    assert env._enemy_objective_zone_cleared(1, _REWARDS) is False


def test_own_army_wipe_still_loss(env):
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=False,
        p1_army_lost=True,
        p2_army_lost=False,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == LOSS


def test_mutual_army_wipe_still_draw(env):
    term, delta = env._pylon_army_outcome(
        p1_lost=False,
        p2_lost=False,
        p1_army_lost=True,
        p2_army_lost=True,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is True
    assert delta == DRAW


def test_own_flag_pylon_loss_alone_does_not_end_episode(env):
    """Flag pylons are anchors only — losing one does not auto-lose."""
    term, delta = env._pylon_army_outcome(
        p1_lost=True,
        p2_lost=False,
        p1_army_lost=False,
        p2_army_lost=False,
        perspective_player_id=1,
        r_conf=_REWARDS,
    )
    assert term is False
    assert delta == 0.0


def test_episode_outcome_flags_on_zone_clear_win(env):
    env._enemy_objective_zone_cleared = MagicMock(return_value=True)
    flags = env._episode_outcome_flags(
        p1_lost=False,
        p2_lost=True,
        p1_army_lost=False,
        p2_army_lost=True,
        all_units_gone=False,
        timeout=False,
        r_conf=_REWARDS,
        perspective_player_id=1,
    )
    assert flags == {"episode_win": 1.0, "episode_loss": 0.0, "episode_draw": 0.0}
