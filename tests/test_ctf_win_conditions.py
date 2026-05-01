"""
Exhaustive unit tests for WinConditionChecker — no SC2 required.
"""
import pytest
from dataclasses import replace
from typing import List

from train.envs.ctf.squad_manager import Outcome, SquadState, SquadStatus
from train.envs.ctf.win_condition import GameState, TerminalResult, WinConditionChecker


CHECKER = WinConditionChecker()
MAX = 1000


def _sq(
    squad_id="sq1",
    role="attack",
    side="player_a",
    status=SquadStatus.ACTIVE,
    assigned_pylon_destroyed=False,
    assigned_attacker_unit_tags=frozenset(),
    alive_attacker_count=None,
) -> SquadState:
    s = SquadState(
        squad_id=squad_id,
        role=role,
        side=side,
        status=status,
        assigned_pylon_destroyed=assigned_pylon_destroyed,
        assigned_attacker_unit_tags=assigned_attacker_unit_tags,
    )
    if alive_attacker_count is not None:
        s._alive_attacker_tags_count = alive_attacker_count
    return s


def _gs(
    step=1,
    learner_pylons=1,
    learner_units=2,
    enemy_pylons=1,
    enemy_units=2,
    squads: List[SquadState] = None,
) -> GameState:
    return GameState(
        step=step,
        max_steps=MAX,
        learner_pylons=learner_pylons,
        learner_units_alive=learner_units,
        enemy_pylons_alive=enemy_pylons,
        enemy_units_alive=enemy_units,
        learner_squads=squads or [],
    )


class TestLossConditions:
    def test_loss_when_pylons_zero(self):
        gs = _gs(learner_pylons=0, learner_units=5)
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.LOSS
        assert result.reason == "learner_wiped"

    def test_loss_when_units_zero(self):
        gs = _gs(learner_pylons=2, learner_units=0)
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.LOSS
        assert result.reason == "learner_wiped"

    def test_loss_both_zero(self):
        gs = _gs(learner_pylons=0, learner_units=0)
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.LOSS

    def test_loss_takes_precedence_over_enemy_dead(self):
        gs = _gs(learner_pylons=0, learner_units=0, enemy_units=0)
        result = CHECKER.check(gs)
        assert result.outcome == Outcome.LOSS


class TestWinA:
    def test_win_a_all_enemy_dead(self):
        gs = _gs(enemy_units=0, squads=[_sq(role="attack")])
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.WIN
        assert result.reason == "all_enemy_dead"

    def test_win_a_requires_enemy_units_zero(self):
        gs = _gs(enemy_units=1, squads=[_sq(role="attack")])
        result = CHECKER.check(gs)
        assert not result.done


class TestWinB:
    def test_win_b_attack_only_pylons_dead(self):
        sq = _sq(role="attack")
        gs = _gs(enemy_units=3, enemy_pylons=0, squads=[sq])
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.WIN
        assert result.reason == "all_enemy_pylons_dead"

    def test_win_b_attack_only_pylons_alive(self):
        sq = _sq(role="attack")
        gs = _gs(enemy_units=3, enemy_pylons=1, squads=[sq])
        result = CHECKER.check(gs)
        assert not result.done

    def test_win_b_does_not_apply_to_defend(self):
        sq = _sq(role="defend")
        gs = _gs(enemy_units=3, enemy_pylons=0, squads=[sq])
        result = CHECKER.check(gs)
        assert not result.done


class TestWinC:
    def test_win_c_timeout_survived(self):
        sq = _sq(role="defend")
        gs = _gs(step=MAX, enemy_units=2, squads=[sq], learner_pylons=1, learner_units=3)
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.WIN
        assert result.reason == "timeout_defend"

    def test_win_c_timeout_dead_pylon(self):
        sq = _sq(role="defend")
        gs = _gs(step=MAX, enemy_units=2, squads=[sq], learner_pylons=0, learner_units=3)
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.LOSS
        assert result.reason == "learner_wiped"

    def test_win_c_immediate_all_enemy_dead(self):
        sq = _sq(role="defend")
        gs = _gs(step=5, enemy_units=0, squads=[sq])
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.WIN
        assert result.reason in ("all_enemy_dead", "all_enemy_dead_defend")

    def test_win_c_ongoing_before_timeout(self):
        sq = _sq(role="defend")
        gs = _gs(step=MAX - 1, enemy_units=2, squads=[sq])
        result = CHECKER.check(gs)
        assert not result.done


class TestWinD:
    def test_win_d_both_objectives_met(self):
        atk = _sq("atk", role="attack", assigned_pylon_destroyed=True)
        def_ = _sq("def", role="defend",
                   assigned_attacker_unit_tags=frozenset({1, 2}),
                   alive_attacker_count=0)
        gs = _gs(enemy_units=1, squads=[atk, def_])
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.WIN
        assert result.reason == "mixed_objectives_met"

    def test_win_d_attack_incomplete(self):
        atk = _sq("atk", role="attack", assigned_pylon_destroyed=False)
        def_ = _sq("def", role="defend",
                   assigned_attacker_unit_tags=frozenset({1}),
                   alive_attacker_count=0)
        gs = _gs(enemy_units=1, squads=[atk, def_])
        result = CHECKER.check(gs)
        assert not result.done

    def test_win_d_defend_incomplete(self):
        atk = _sq("atk", role="attack", assigned_pylon_destroyed=True)
        def_ = _sq("def", role="defend",
                   assigned_attacker_unit_tags=frozenset({1, 2}),
                   alive_attacker_count=1)
        gs = _gs(enemy_units=1, squads=[atk, def_])
        result = CHECKER.check(gs)
        assert not result.done

    def test_win_d_active_attacker_pylon_done_and_defender_done(self):
        atk = _sq("atk", role="attack", assigned_pylon_destroyed=True)
        def_ = _sq("def", role="defend",
                   assigned_attacker_unit_tags=frozenset({1, 2}),
                   alive_attacker_count=0)
        gs = _gs(enemy_units=1, squads=[atk, def_])
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.WIN
        assert result.reason == "mixed_objectives_met"

    def test_ghost_only_attackers_gives_defend_only_composition(self):
        atk = _sq("atk", role="attack", status=SquadStatus.GHOST)
        def_ = _sq("def", role="defend")
        gs = _gs(step=1, enemy_units=2, squads=[atk, def_])
        result = CHECKER.check(gs)
        assert not result.done


class TestTimeout:
    def test_timeout_non_defend_is_loss(self):
        atk = _sq(role="attack", assigned_pylon_destroyed=False)
        gs = _gs(step=MAX, enemy_units=3, enemy_pylons=1, squads=[atk])
        result = CHECKER.check(gs)
        assert result.done
        assert result.outcome == Outcome.LOSS
        assert result.reason == "timeout_non_defend"

    def test_no_timeout_before_max_steps(self):
        atk = _sq(role="attack")
        gs = _gs(step=MAX - 1, enemy_units=3, enemy_pylons=1, squads=[atk])
        result = CHECKER.check(gs)
        assert not result.done

    def test_ongoing_returns_done_false(self):
        sq = _sq(role="attack")
        gs = _gs(enemy_units=2, enemy_pylons=1, squads=[sq])
        result = CHECKER.check(gs)
        assert not result.done
        assert result.outcome == Outcome.ONGOING
