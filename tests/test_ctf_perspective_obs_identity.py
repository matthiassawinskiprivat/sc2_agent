"""P1 trainee and P2 trainee must see the same world-frame observation.

CTF spawn is region/role, not ladder home corners. After the leak-fix, Terran-as-P2
sits in the same cells Terran-as-P1 used to. Allied/enemy is owner-id only — no 180°
map rotation. This test swaps owner ids while keeping world positions fixed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from src.agent.squad_constants import encode_race_vector
from src.transformers.squad_observation_builder import SquadObservationBuilder
from tests.test_observation_transformer import _make_game_info, create_mock_unit


class _UnitsList(list):
    """Iterable stand-in for python-sc2 Units, with find_by_tag."""

    @property
    def exists(self):
        return len(self) > 0

    @property
    def first(self):
        return self[0] if self else None

    def __call__(self, type_id):
        return _UnitsList(u for u in self if getattr(u, "type_id", None) == type_id)

    def find_by_tag(self, tag):
        for unit in self:
            if getattr(unit, "tag", None) == tag:
                return unit
        return None


def _combat_unit(type_id, pos, *, owner_id, tag, **kwargs):
    unit = create_mock_unit(type_id, pos, radius=0.375, **kwargs)
    unit.tag = tag
    unit.owner_id = owner_id
    unit.is_structure = False
    return unit


def _pylon(pos, *, owner_id, tag):
    unit = create_mock_unit(UnitTypeId.PYLON, pos, radius=1.0)
    unit.tag = tag
    unit.owner_id = owner_id
    unit.is_structure = True
    return unit


def _god_mode_bot(mobiles, structures):
    """Both armies in P1-client buckets (debug_control_enemy), split by owner_id."""
    bot = MagicMock()
    bot.player_id = 1
    bot.units = _UnitsList(mobiles)
    bot.enemy_units = _UnitsList()
    bot.structures = _UnitsList(structures)
    bot.enemy_structures = _UnitsList()
    all_units = _UnitsList(list(mobiles) + list(structures))
    bot.all_units = all_units
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.upgrades = frozenset()
    bot.game_info = _make_game_info(128, 128, with_grids=True)
    bot.game_info.playable_area = MagicMock()
    bot.game_info.playable_area.x = 0
    bot.game_info.playable_area.y = 0
    bot.game_info.playable_area.width = 128
    bot.game_info.playable_area.height = 128
    bot.game_data = MagicMock()
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.unit_target_map = {}
    bot.unit_command_map = {}
    return bot


def _training_conf():
    return {
        "agent_architecture": "squad",
        "include_race": True,
        "active_layers": [
            "allied_unit_density",
            "enemy_unit_density",
            "command_objective",
        ],
        "active_tactical_layers": [
            "allied_unit_density",
            "enemy_unit_density",
            "command_objective",
        ],
        "active_unit_scalars": [
            "x_norm",
            "y_norm",
            "dist_to_objective_norm",
        ],
    }


def _build_for_owner(trainee_owner: int):
    """Trainee marine at (30, 30), enemy zergling at (90, 90), shared objective."""
    enemy_owner = 2 if trainee_owner == 1 else 1
    own_pos = Point2((30.0, 30.0))
    enemy_pos = Point2((90.0, 90.0))
    objective = Point2((80.0, 40.0))

    marine = _combat_unit(UnitTypeId.MARINE, own_pos, owner_id=trainee_owner, tag=10)
    zergling = _combat_unit(UnitTypeId.ZERGLING, enemy_pos, owner_id=enemy_owner, tag=20)
    own_pylon = _pylon(Point2((25.0, 25.0)), owner_id=trainee_owner, tag=101)
    enemy_pylon = _pylon(Point2((95.0, 95.0)), owner_id=enemy_owner, tag=201)

    bot = _god_mode_bot([marine, zergling], [own_pylon, enemy_pylon])
    bot.unit_target_map = {marine.tag: objective}
    bot.unit_command_map = {marine.tag: "attack"}

    builder = SquadObservationBuilder(_training_conf())
    obs = builder.build(
        bot,
        perspective_player_id=trainee_owner,
        squad_unit_tags=[marine.tag],
        pool_high_level_command="attack",
        playable_size=(128.0, 128.0),
        anchor_unit_tag=marine.tag,
        own_race="terran",
        enemy_race="zerg",
    )
    return obs


def test_p1_and_p2_trainee_obs_match_with_swapped_owners():
    obs_p1 = _build_for_owner(1)
    obs_p2 = _build_for_owner(2)

    layers = _training_conf()["active_layers"]
    allied_idx = layers.index("allied_unit_density")
    enemy_idx = layers.index("enemy_unit_density")
    obj_idx = layers.index("command_objective")

    assert np.allclose(obs_p1["map"][allied_idx], obs_p2["map"][allied_idx])
    assert np.allclose(obs_p1["map"][enemy_idx], obs_p2["map"][enemy_idx])
    assert np.allclose(obs_p1["map"][obj_idx], obs_p2["map"][obj_idx])
    assert float(obs_p1["map"][obj_idx].sum()) > 0.0
    assert float(obs_p1["map"][allied_idx].sum()) > 0.0
    assert float(obs_p1["map"][enemy_idx].sum()) > 0.0
    assert not np.allclose(obs_p1["map"][allied_idx], obs_p1["map"][enemy_idx])

    assert np.allclose(obs_p1["unit_scalars"][0], obs_p2["unit_scalars"][0])
    assert abs(float(obs_p1["unit_scalars"][0, 0]) - 30.0 / 128.0) < 1e-5
    assert abs(float(obs_p1["unit_scalars"][0, 1]) - 30.0 / 128.0) < 1e-5

    expected_race = encode_race_vector("terran", "zerg")
    assert np.allclose(obs_p1["race"], expected_race)
    assert np.allclose(obs_p2["race"], expected_race)

    assert np.allclose(obs_p1["command"], obs_p2["command"])
    assert np.allclose(obs_p1["anchor_tactical"], obs_p2["anchor_tactical"])
    assert float(obs_p1["anchor_tactical"].sum()) > 0.0


def _attach_bot_object(units, bot):
    for unit in units:
        unit._bot_object = bot
        unit.is_mine = getattr(unit, "owner_id", 0) == 1


def _dps_training_conf():
    conf = _training_conf()
    conf["active_layers"] = [
        "allied_unit_density",
        "enemy_unit_density",
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "command_objective",
    ]
    conf["active_tactical_layers"] = [
        "allied_unit_density",
        "enemy_unit_density",
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "command_objective",
    ]
    return conf


def _build_hydra_dps_for_owner(trainee_owner: int, *, with_upgrade: bool):
    """Trainee hydralisk at (30, 30); P1-client is_mine only on owner 1."""
    enemy_owner = 2 if trainee_owner == 1 else 1
    hydra = _combat_unit(
        UnitTypeId.HYDRALISK,
        Point2((30.0, 30.0)),
        owner_id=trainee_owner,
        tag=10,
        ground_range=5.0,
        air_range=5.0,
    )
    marine = _combat_unit(
        UnitTypeId.MARINE,
        Point2((90.0, 90.0)),
        owner_id=enemy_owner,
        tag=20,
        ground_range=5.0,
        air_range=0.0,
    )
    own_pylon = _pylon(Point2((25.0, 25.0)), owner_id=trainee_owner, tag=101)
    enemy_pylon = _pylon(Point2((95.0, 95.0)), owner_id=enemy_owner, tag=201)
    bot = _god_mode_bot([hydra, marine], [own_pylon, enemy_pylon])
    bot.state.upgrades = (
        frozenset({UpgradeId.EVOLVEGROOVEDSPINES}) if with_upgrade else frozenset()
    )
    _attach_bot_object([hydra, marine, own_pylon, enemy_pylon], bot)
    bot.unit_target_map = {hydra.tag: Point2((80.0, 40.0))}
    bot.unit_command_map = {hydra.tag: "attack"}
    builder = SquadObservationBuilder(_dps_training_conf())
    return builder.build(
        bot,
        perspective_player_id=trainee_owner,
        squad_unit_tags=[hydra.tag],
        pool_high_level_command="attack",
        playable_size=(128.0, 128.0),
        anchor_unit_tag=hydra.tag,
        own_race="zerg",
        enemy_race="terran",
    )


def test_p1_and_p2_trainee_dps_rings_match_with_global_upgrades():
    """P2 hydralisk is_mine=False must still get Grooved Spines DPS reach."""
    obs_p1 = _build_hydra_dps_for_owner(1, with_upgrade=True)
    obs_p2 = _build_hydra_dps_for_owner(2, with_upgrade=True)
    obs_p2_bare = _build_hydra_dps_for_owner(2, with_upgrade=False)

    layers = _dps_training_conf()["active_layers"]
    allied_dps = layers.index("allied_g_dps_base")
    enemy_dps = layers.index("enemy_g_dps_base")
    tac_layers = _dps_training_conf()["active_tactical_layers"]
    tac_allied_dps = tac_layers.index("allied_g_dps_base")

    assert np.allclose(obs_p1["map"][allied_dps], obs_p2["map"][allied_dps])
    assert np.allclose(obs_p1["map"][enemy_dps], obs_p2["map"][enemy_dps])
    assert np.allclose(obs_p1["anchor_tactical"][tac_allied_dps], obs_p2["anchor_tactical"][tac_allied_dps])
    assert float(obs_p2["map"][allied_dps].sum()) > float(obs_p2_bare["map"][allied_dps].sum())
    assert float(obs_p1["anchor_tactical"][tac_allied_dps].sum()) > 0.0


def test_p2_trainee_paints_p1_cloaked_enemy_covered_by_own_observer():
    """Proto is_detected=False on P1 cloaked units must not hide them from P2."""
    enemy_owner = 1
    trainee_owner = 2
    observer = _combat_unit(
        UnitTypeId.OBSERVER,
        Point2((30.0, 30.0)),
        owner_id=trainee_owner,
        tag=10,
        detect_range=11.0,
        is_flying=True,
    )
    dt = _combat_unit(
        UnitTypeId.DARKTEMPLAR,
        Point2((34.0, 30.0)),
        owner_id=enemy_owner,
        tag=20,
        is_cloaked=True,
        is_detected=False,
        is_snapshot=False,
    )
    own_pylon = _pylon(Point2((25.0, 25.0)), owner_id=trainee_owner, tag=101)
    enemy_pylon = _pylon(Point2((95.0, 95.0)), owner_id=enemy_owner, tag=201)
    bot = _god_mode_bot([observer, dt], [own_pylon, enemy_pylon])
    bot.unit_target_map = {observer.tag: Point2((80.0, 40.0))}
    bot.unit_command_map = {observer.tag: "attack"}
    builder = SquadObservationBuilder(_training_conf())
    obs = builder.build(
        bot,
        perspective_player_id=trainee_owner,
        squad_unit_tags=[observer.tag],
        pool_high_level_command="attack",
        playable_size=(128.0, 128.0),
        anchor_unit_tag=observer.tag,
        own_race="protoss",
        enemy_race="protoss",
    )
    layers = _training_conf()["active_layers"]
    enemy_idx = layers.index("enemy_unit_density")
    tac_enemy = _training_conf()["active_tactical_layers"].index("enemy_unit_density")
    assert float(obs["map"][enemy_idx].sum()) > 0.0
    assert float(obs["anchor_tactical"][tac_enemy].sum()) > 0.0


def test_p1_and_p2_trainee_cloaked_enemy_obs_match_with_own_detector():
    """Owner-swap identity: cloaked enemy in detector range, proto undetected."""

    def _build(trainee_owner: int):
        enemy_owner = 2 if trainee_owner == 1 else 1
        observer = _combat_unit(
            UnitTypeId.OBSERVER,
            Point2((30.0, 30.0)),
            owner_id=trainee_owner,
            tag=10,
            detect_range=11.0,
            is_flying=True,
        )
        dt = _combat_unit(
            UnitTypeId.DARKTEMPLAR,
            Point2((34.0, 30.0)),
            owner_id=enemy_owner,
            tag=20,
            is_cloaked=True,
            is_detected=False,
            is_snapshot=False,
        )
        own_pylon = _pylon(Point2((25.0, 25.0)), owner_id=trainee_owner, tag=101)
        enemy_pylon = _pylon(Point2((95.0, 95.0)), owner_id=enemy_owner, tag=201)
        bot = _god_mode_bot([observer, dt], [own_pylon, enemy_pylon])
        bot.unit_target_map = {observer.tag: Point2((80.0, 40.0))}
        bot.unit_command_map = {observer.tag: "attack"}
        builder = SquadObservationBuilder(_training_conf())
        return builder.build(
            bot,
            perspective_player_id=trainee_owner,
            squad_unit_tags=[observer.tag],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
            anchor_unit_tag=observer.tag,
            own_race="protoss",
            enemy_race="protoss",
        )

    obs_p1 = _build(1)
    obs_p2 = _build(2)
    layers = _training_conf()["active_layers"]
    enemy_idx = layers.index("enemy_unit_density")
    allied_idx = layers.index("allied_unit_density")
    assert np.allclose(obs_p1["map"][enemy_idx], obs_p2["map"][enemy_idx])
    assert np.allclose(obs_p1["map"][allied_idx], obs_p2["map"][allied_idx])
    assert np.allclose(obs_p1["anchor_tactical"], obs_p2["anchor_tactical"])
    assert float(obs_p1["map"][enemy_idx].sum()) > 0.0
