"""
Unit tests for SquadManager reassignment logic and round-robin.
"""
import pytest
from typing import Dict, List

from train.envs.ctf.config import BudgetConfig, CTFConfig, SquadConfig
from train.envs.ctf.region_loader import CircleRegion, Point2, RectRegion
from train.envs.ctf.squad_manager import (
    GameEvent,
    PylonDeathEvent,
    SquadManager,
    SquadStatus,
    UnitDeathEvent,
    compare_id,
)
from train.envs.ctf.game_data import uid


def _regions(centers: Dict[str, tuple]) -> Dict[str, CircleRegion]:
    return {
        name: CircleRegion(name=name, center=Point2(*xy), radius=5.0)
        for name, xy in centers.items()
    }


def _std_config():
    return CTFConfig(
        map_path="dummy.sc2map",
        max_steps=1000,
        every_k_episodes=50,
        squads=[
            SquadConfig(
                id="atk_a", role="attack", side="player_a",
                budget=BudgetConfig(minerals=200, vespene=0),
                spawn_pool=[uid("MARINE")], spawn_region="base_a",
                attack_targets=["obj_b"],
            ),
            SquadConfig(
                id="def_a", role="defend", side="player_a",
                budget=BudgetConfig(minerals=200, vespene=0),
                spawn_pool=[uid("MARINE")], spawn_region="base_a",
                defend_anchors=["obj_a"],
            ),
            SquadConfig(
                id="atk_b", role="attack", side="player_b",
                budget=BudgetConfig(minerals=200, vespene=0),
                spawn_pool=[uid("MARINE")], spawn_region="base_b",
                attack_targets=["obj_a"],
            ),
            SquadConfig(
                id="def_b", role="defend", side="player_b",
                budget=BudgetConfig(minerals=200, vespene=0),
                spawn_pool=[uid("MARINE")], spawn_region="base_b",
                defend_anchors=["obj_b"],
            ),
        ],
    )


def _make_sm(regions_centers=None) -> SquadManager:
    if regions_centers is None:
        regions_centers = {
            "base_a": (0.0, 0.0),
            "obj_a": (10.0, 0.0),
            "base_b": (100.0, 0.0),
            "obj_b": (90.0, 0.0),
        }
    cfg = _std_config()
    regs = _regions(regions_centers)
    sm = SquadManager(squad_configs=cfg.squads, learner_side="player_a", regions=regs)
    sm.register_unit_tags("atk_a", [10, 11, 12])
    sm.register_unit_tags("def_a", [20, 21])
    sm.register_unit_tags("atk_b", [30, 31])
    sm.register_unit_tags("def_b", [40, 41])
    return sm


class TestCompareId:
    def test_numeric_lower(self):
        assert compare_id("2", "10") < 0

    def test_numeric_equal(self):
        assert compare_id("5", "5") == 0

    def test_numeric_higher(self):
        assert compare_id("10", "2") > 0

    def test_string_lower(self):
        assert compare_id("alpha", "beta") < 0

    def test_string_equal(self):
        assert compare_id("foo", "foo") == 0

    def test_numeric_beats_string(self):
        assert compare_id("5", "abc") < 0

    def test_mixed_zero_vs_string(self):
        assert compare_id("0", "abc") < 0


class TestAttackerDoneMerge:
    def test_attacker_done_transfers_units_to_ally(self):
        sm = _make_sm()
        atk_state = sm.get_state("atk_a")
        def_state = sm.get_state("def_a")
        initial_def_count = def_state.unit_count

        atk_state.assigned_pylon_destroyed = True
        sm.check_and_reassign(sm._regions)

        assert atk_state.status == SquadStatus.GHOST
        assert atk_state.unit_count == 0
        assert def_state.unit_count == initial_def_count + 3

    def test_attacker_becomes_ghost(self):
        sm = _make_sm()
        sm.get_state("atk_a").assigned_pylon_destroyed = True
        sm.check_and_reassign(sm._regions)
        assert sm.get_state("atk_a").status == SquadStatus.GHOST

    def test_merged_units_tag_map_updated(self):
        sm = _make_sm()
        sm.get_state("atk_a").assigned_pylon_destroyed = True
        sm.check_and_reassign(sm._regions)
        for tag in [10, 11, 12]:
            assert sm.unit_to_squad(tag) == "def_a"

    def test_closest_ally_chosen(self):
        regions = {
            "base_a": (0.0, 0.0),
            "obj_a": (10.0, 0.0),
            "obj_a2": (80.0, 0.0),
            "base_b": (100.0, 0.0),
            "obj_b": (90.0, 0.0),
        }
        cfg = CTFConfig(
            map_path="dummy.sc2map",
            max_steps=1000,
            every_k_episodes=50,
            squads=[
                SquadConfig(id="atk_a", role="attack", side="player_a",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_a",
                            attack_targets=["obj_b"]),
                SquadConfig(id="def_close", role="defend", side="player_a",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_a",
                            defend_anchors=["obj_a"]),
                SquadConfig(id="def_far", role="defend", side="player_a",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_a",
                            defend_anchors=["obj_a2"]),
                SquadConfig(id="def_b", role="defend", side="player_b",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_b",
                            defend_anchors=["obj_b"]),
                SquadConfig(id="atk_b", role="attack", side="player_b",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_b",
                            attack_targets=["obj_a"]),
            ],
        )
        reg_defs = {n: CircleRegion(name=n, center=Point2(*xy), radius=5.0)
                    for n, xy in regions.items()}
        sm = SquadManager(cfg.squads, learner_side="player_a", regions=reg_defs)
        sm.register_unit_tags("atk_a", [1, 2])
        sm.register_unit_tags("def_close", [3])
        sm.register_unit_tags("def_far", [4])
        sm.register_unit_tags("def_b", [5])
        sm.register_unit_tags("atk_b", [6])

        sm.get_state("atk_a").assigned_pylon_destroyed = True
        sm.check_and_reassign(reg_defs)

        assert sm.get_state("def_far").unit_count == 1 + 2


class TestTieBreak:
    def test_tie_break_lower_id_wins(self):
        regions = {
            "obj_b": (50.0, 0.0),
            "anchor_1": (50.0, 10.0),
            "anchor_2": (50.0, -10.0),
            "base_a": (0.0, 0.0),
            "base_b": (100.0, 0.0),
        }
        cfg = CTFConfig(
            map_path="dummy.sc2map",
            max_steps=1000,
            every_k_episodes=50,
            squads=[
                SquadConfig(id="atk_a", role="attack", side="player_a",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_a",
                            attack_targets=["obj_b"]),
                SquadConfig(id="def_1", role="defend", side="player_a",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_a",
                            defend_anchors=["anchor_1"]),
                SquadConfig(id="def_2", role="defend", side="player_a",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_a",
                            defend_anchors=["anchor_2"]),
                SquadConfig(id="def_b", role="defend", side="player_b",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_b",
                            defend_anchors=["obj_b"]),
                SquadConfig(id="atk_b", role="attack", side="player_b",
                            budget=BudgetConfig(minerals=100, vespene=0),
                            spawn_pool=[uid("MARINE")], spawn_region="base_b",
                            attack_targets=["anchor_1"]),
            ],
        )
        reg_defs = {n: CircleRegion(n, Point2(*xy), 5.0) for n, xy in regions.items()}
        sm = SquadManager(cfg.squads, learner_side="player_a", regions=reg_defs)
        sm.register_unit_tags("atk_a", [10, 11])
        sm.register_unit_tags("def_1", [20])
        sm.register_unit_tags("def_2", [30])
        sm.register_unit_tags("def_b", [40])
        sm.register_unit_tags("atk_b", [50])

        sm.get_state("atk_a").assigned_pylon_destroyed = True
        sm.check_and_reassign(reg_defs)

        assert sm.get_state("def_1").unit_count == 1 + 2
        assert sm.get_state("def_2").unit_count == 1


class TestAttackerEliminatedDefenderReassign:
    def test_eliminated_attacker_triggers_defender(self):
        sm = _make_sm()
        atk_state = sm.get_state("atk_a")
        def_b_state = sm.get_state("def_b")
        initial_def_b_units = def_b_state.unit_count

        atk_state.unit_tags.clear()
        atk_state.status = SquadStatus.ELIMINATED

        sm.check_and_reassign(sm._regions)

        assert def_b_state.status == SquadStatus.GHOST
        assert def_b_state.unit_count == 0

        atk_b_state = sm.get_state("atk_b")
        assert atk_b_state.unit_count == 2 + initial_def_b_units


class TestRoundRobinGhostSkip:
    def test_rr_skips_ghost(self):
        sm = _make_sm()
        sm.get_state("atk_a").status = SquadStatus.GHOST
        sm._learner_rr_idx = 0
        next_sq = sm.advance_learner_rr()
        assert next_sq is not None
        assert next_sq.squad_id == "def_a"

    def test_rr_returns_none_when_all_ghost(self):
        sm = _make_sm()
        for sid in ["atk_a", "def_a"]:
            sm.get_state(sid).status = SquadStatus.GHOST
        result = sm.advance_learner_rr()
        assert result is None

    def test_rr_wraps_around(self):
        sm = _make_sm()
        sq1 = sm.advance_learner_rr()
        sq2 = sm.advance_learner_rr()
        sq3 = sm.advance_learner_rr()
        assert sq3 is not None
        assert sq3.squad_id == sq1.squad_id


class TestUnitCountInvariants:
    def test_total_units_preserved_after_merge(self):
        sm = _make_sm()
        total_before = sm.learner_unit_count() + sum(
            sm.get_state(s).unit_count for s in ["atk_b", "def_b"]
        )
        sm.get_state("atk_a").assigned_pylon_destroyed = True
        sm.check_and_reassign(sm._regions)
        total_after = sm.learner_unit_count() + sum(
            sm.get_state(s).unit_count for s in ["atk_b", "def_b"]
        )
        assert total_before == total_after

    def test_no_orphan_tags_after_merge(self):
        sm = _make_sm()
        sm.get_state("atk_a").assigned_pylon_destroyed = True
        sm.check_and_reassign(sm._regions)
        all_tags_in_squads = set()
        for state in sm._states.values():
            all_tags_in_squads.update(state.unit_tags)
        for tag in [10, 11, 12, 20, 21]:
            assert tag in all_tags_in_squads or tag in sm._tag_to_squad
