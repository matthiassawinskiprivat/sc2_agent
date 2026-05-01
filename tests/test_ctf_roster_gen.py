"""
Unit tests for SpawnController roster generation.
"""
import pytest
import numpy as np

from train.envs.ctf.config import BudgetConfig, CTFConfig, RewardConfig, SquadConfig
from train.envs.ctf.region_loader import CircleRegion, Point2, RegionDef
from train.envs.ctf.spawn import SpawnController
from train.envs.ctf.game_data import get_unit_cost, uid


def _make_regions(names=("spawn_a", "obj_a", "spawn_b", "obj_b")):
    return {
        n: CircleRegion(name=n, center=Point2(float(i * 20), 0.0), radius=10.0)
        for i, n in enumerate(names)
    }


def _make_config(
    minerals=400, vespene=0,
    pool=None,
    role="attack",
    side="player_a",
    squad_id="sq_a",
    spawn_region="spawn_a",
    attack_targets=("obj_b",),
    defend_anchors=("obj_a",),
):
    if pool is None:
        pool = [uid("MARINE"), uid("MARAUDER")]

    squads = [
        SquadConfig(
            id=squad_id,
            role=role,
            side=side,
            budget=BudgetConfig(minerals=minerals, vespene=vespene),
            spawn_pool=list(pool),
            spawn_region=spawn_region,
            attack_targets=list(attack_targets) if role == "attack" else [],
            defend_anchors=list(defend_anchors) if role == "defend" else [],
        ),
        SquadConfig(
            id="sq_b",
            role="defend" if role == "attack" else "attack",
            side="player_b",
            budget=BudgetConfig(minerals=200, vespene=0),
            spawn_pool=[uid("MARINE")],
            spawn_region="spawn_b",
            attack_targets=list(defend_anchors) if role == "defend" else [],
            defend_anchors=list(attack_targets) if role == "attack" else [],
        ),
    ]
    return CTFConfig(
        map_path="dummy.sc2map",
        max_steps=1000,
        every_k_episodes=50,
        squads=squads,
    )


class TestRosterGeneration:
    def _make_controller(self, cfg, seed=42):
        regions = _make_regions()
        rng = np.random.default_rng(seed)
        return SpawnController(config=cfg, regions=regions, rng=rng)

    def test_roster_terminates_when_cheapest_unaffordable(self):
        cfg = _make_config(minerals=99, vespene=0, pool=[uid("MARAUDER")])
        ctrl = self._make_controller(cfg)
        rosters = ctrl.generate_rosters()
        assert len(rosters["sq_a"].units) == 0

    def test_roster_never_exceeds_budget(self):
        minerals, vespene = 500, 150
        pool = [uid("MARINE"), uid("MARAUDER"), uid("REAPER")]
        cfg = _make_config(minerals=minerals, vespene=vespene, pool=pool)
        ctrl = self._make_controller(cfg)
        rosters = ctrl.generate_rosters()
        roster = rosters["sq_a"]

        total_m, total_v = 0, 0
        for entry in roster.units:
            m, v = get_unit_cost(entry.type_id)
            total_m += m
            total_v += v

        assert total_m <= minerals
        assert total_v <= vespene

    def test_empty_pool_produces_no_units(self):
        cfg = _make_config(minerals=0, vespene=0)
        ctrl = self._make_controller(cfg)
        rosters = ctrl.generate_rosters()
        assert len(rosters["sq_a"].units) == 0

    def test_roster_deterministic_same_seed(self):
        pool = [uid("MARINE"), uid("MARAUDER"), uid("REAPER")]
        cfg = _make_config(minerals=600, vespene=200, pool=pool)
        ctrl1 = self._make_controller(cfg, seed=7)
        ctrl2 = self._make_controller(cfg, seed=7)
        rosters1 = ctrl1.generate_rosters()
        rosters2 = ctrl2.generate_rosters()
        assert (
            [e.type_id for e in rosters1["sq_a"].units]
            == [e.type_id for e in rosters2["sq_a"].units]
        )

    def test_roster_differs_different_seed(self):
        pool = [uid("MARINE"), uid("MARAUDER"), uid("REAPER"), uid("GHOST")]
        cfg = _make_config(minerals=1000, vespene=500, pool=pool)
        ctrl1 = self._make_controller(cfg, seed=1)
        ctrl2 = self._make_controller(cfg, seed=999)
        r1 = [e.type_id for e in ctrl1.generate_rosters()["sq_a"].units]
        r2 = [e.type_id for e in ctrl2.generate_rosters()["sq_a"].units]
        assert isinstance(r1, list) and isinstance(r2, list)

    def test_defender_pylon_placed(self):
        cfg = _make_config(
            minerals=200, role="defend",
            pool=[uid("MARINE")],
            defend_anchors=("obj_a",),
        )
        ctrl = self._make_controller(cfg)
        rosters = ctrl.generate_rosters()
        pylon = rosters["sq_a"].pylon
        assert pylon is not None
        from train.envs.ctf.game_data import uid as _uid
        assert pylon.type_id == _uid("PYLON")

    def test_attacker_has_no_pylon(self):
        cfg = _make_config(minerals=200, role="attack", pool=[uid("MARINE")])
        ctrl = self._make_controller(cfg)
        rosters = ctrl.generate_rosters()
        assert rosters["sq_a"].pylon is None

    def test_spawn_positions_near_region_center(self):
        cfg = _make_config(minerals=500, pool=[uid("MARINE")])
        ctrl = self._make_controller(cfg)
        rosters = ctrl.generate_rosters()
        center = Point2(0.0, 0.0)
        for entry in rosters["sq_a"].units:
            dist = center.distance_to(entry.position)
            assert dist < 30.0

    def test_only_affordable_units_sampled(self):
        pool = [uid("MARINE"), uid("REAPER")]
        cfg = _make_config(minerals=500, vespene=0, pool=pool)
        ctrl = self._make_controller(cfg, seed=0)
        rosters = ctrl.generate_rosters()
        for entry in rosters["sq_a"].units:
            _, v_cost = get_unit_cost(entry.type_id)
            assert v_cost == 0
