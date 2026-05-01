"""
Unit tests for RoleRewardComputer.
"""
import pytest

from train.envs.ctf.config import (
    AttackerRewardConfig,
    BudgetConfig,
    CTFConfig,
    DefenderRewardConfig,
    RewardConfig,
    SquadConfig,
)
from train.envs.ctf.game_data import uid
from train.envs.ctf.reward import (
    PylonDestroyRewardEvent,
    RoleRewardComputer,
    StepEvents,
    UnitKillRewardEvent,
    WeaponFireRewardEvent,
)
from train.envs.ctf.squad_manager import Outcome, SquadState, SquadStatus
from train.envs.ctf.win_condition import GameState


def _make_reward_config(
    atk_step=-0.01, atk_pylon=10.0, atk_kill=1.0,
    atk_ttb=2.0, atk_ttb_clip=5.0,
    def_step=0.01, def_pylon=5.0, def_kill=1.0,
    def_ttb=2.0, def_ttb_clip=5.0,
) -> RewardConfig:
    return RewardConfig(
        attacker=AttackerRewardConfig(
            step_penalty=atk_step,
            pylon_destroy_bonus=atk_pylon,
            unit_kill_bonus=atk_kill,
            target_type_bonus=atk_ttb,
            target_type_bonus_clip=atk_ttb_clip,
        ),
        defender=DefenderRewardConfig(
            step_reward=def_step,
            pylon_survival_bonus=def_pylon,
            unit_kill_bonus=def_kill,
            target_type_bonus=def_ttb,
            target_type_bonus_clip=def_ttb_clip,
        ),
    )


def _sq(role="attack", status=SquadStatus.ACTIVE, squad_id="sq1") -> SquadState:
    return SquadState(squad_id=squad_id, role=role, side="player_a", status=status)


def _gs(step=1) -> GameState:
    return GameState(
        step=step, max_steps=1000,
        learner_pylons=1, learner_units_alive=2,
        enemy_pylons_alive=1, enemy_units_alive=2,
    )


def _make_computer(**kwargs) -> RoleRewardComputer:
    return RoleRewardComputer(_make_reward_config(**kwargs))


class TestAttackerRewards:
    def test_step_penalty_applied(self):
        cmp = _make_computer(atk_step=-0.01)
        r = cmp.compute(_sq(role="attack"), StepEvents(), _gs())
        assert abs(r - (-0.01)) < 1e-9

    def test_pylon_destroy_bonus(self):
        cmp = _make_computer(atk_step=0.0, atk_pylon=10.0, atk_kill=0.0)
        events = StepEvents(pylon_destroy=[PylonDestroyRewardEvent(pylon_tag=1)])
        r = cmp.compute(_sq(role="attack"), events, _gs())
        assert abs(r - 10.0) < 1e-9

    def test_unit_kill_bonus(self):
        cmp = _make_computer(atk_step=0.0, atk_pylon=0.0, atk_kill=1.0)
        events = StepEvents(unit_kills=[
            UnitKillRewardEvent(victim_tag=1, victim_type_id=uid("MARINE")),
            UnitKillRewardEvent(victim_tag=2, victim_type_id=uid("MARINE")),
        ])
        r = cmp.compute(_sq(role="attack"), events, _gs())
        assert abs(r - 2.0) < 1e-9

    def test_multiple_bonuses_summed(self):
        cmp = _make_computer(atk_step=-0.001, atk_pylon=10.0, atk_kill=1.0, atk_ttb=0.0)
        events = StepEvents(
            pylon_destroy=[PylonDestroyRewardEvent(pylon_tag=1)],
            unit_kills=[UnitKillRewardEvent(1, uid("MARINE"))],
        )
        r = cmp.compute(_sq(role="attack"), events, _gs())
        assert abs(r - 10.999) < 1e-9


class TestDefenderRewards:
    def test_step_reward_applied(self):
        cmp = _make_computer(def_step=0.01)
        r = cmp.compute(_sq(role="defend"), StepEvents(), _gs())
        assert abs(r - 0.01) < 1e-9

    def test_unit_kill_bonus_defender(self):
        cmp = _make_computer(def_step=0.0, def_kill=2.0)
        events = StepEvents(unit_kills=[UnitKillRewardEvent(1, uid("ZERGLING"))])
        r = cmp.compute(_sq(role="defend"), events, _gs())
        assert abs(r - 2.0) < 1e-9


class TestTargetTypeBonus:
    def test_target_type_bonus_applied(self):
        cmp = _make_computer(atk_step=0.0, atk_ttb=1.0, atk_ttb_clip=100.0)
        events = StepEvents(weapon_fires=[
            WeaponFireRewardEvent(
                attacker_tag=1,
                attacker_type_id=uid("HELLION"),
                target_tag=2,
                target_type_id=uid("MARINE"),
            )
        ])
        r = cmp.compute(_sq(role="attack"), events, _gs())
        assert r > 0.0

    def test_target_type_bonus_clipped(self):
        clip = 3.0
        cmp = _make_computer(atk_step=0.0, atk_ttb=1.0, atk_ttb_clip=clip)
        events = StepEvents(weapon_fires=[
            WeaponFireRewardEvent(1, uid("HELLION"), i, uid("MARINE"))
            for i in range(10)
        ])
        r = cmp.compute(_sq(role="attack"), events, _gs())
        assert r <= clip + 1e-9

    def test_no_bonus_for_no_bonus_matchup(self):
        cmp = _make_computer(atk_step=0.0, atk_ttb=1.0, atk_ttb_clip=100.0)
        events = StepEvents(weapon_fires=[
            WeaponFireRewardEvent(1, uid("MARINE"), 2, uid("MARINE"))
        ])
        r = cmp.compute(_sq(role="attack"), events, _gs())
        assert abs(r) < 1e-9

    def test_clip_value_never_exceeded(self):
        for scalar in [0.5, 1.0, 5.0]:
            for clip in [1.0, 5.0, 10.0]:
                cmp = _make_computer(
                    atk_step=0.0, atk_kill=0.0, atk_ttb=scalar, atk_ttb_clip=clip
                )
                events = StepEvents(weapon_fires=[
                    WeaponFireRewardEvent(1, uid("HELLION"), i, uid("MARINE"))
                    for i in range(20)
                ])
                r = cmp.compute(_sq(role="attack"), events, _gs())
                assert r <= clip + 1e-9


class TestRoleChangeAfterMerge:
    def test_merged_attacker_into_defender_uses_defender_block(self):
        cmp = _make_computer(atk_step=-10.0, def_step=0.01)
        sq = _sq(role="defend")
        r = cmp.compute(sq, StepEvents(), _gs())
        assert abs(r - 0.01) < 1e-9

    def test_attacker_block_for_attack_role(self):
        cmp = _make_computer(atk_step=-0.5, def_step=0.5)
        r = cmp.compute(_sq(role="attack"), StepEvents(), _gs())
        assert r < 0


class TestTerminalBonus:
    def test_defender_pylon_survival_bonus_when_alive(self):
        cmp = _make_computer(def_pylon=7.0)
        sq = _sq(role="defend")
        sq.assigned_pylon_destroyed = False
        bonus = cmp.episode_terminal_bonus(sq, _gs())
        assert abs(bonus - 7.0) < 1e-9

    def test_no_pylon_bonus_when_destroyed(self):
        cmp = _make_computer(def_pylon=7.0)
        sq = _sq(role="defend")
        sq.assigned_pylon_destroyed = True
        assert abs(cmp.episode_terminal_bonus(sq, _gs())) < 1e-9

    def test_no_terminal_bonus_for_attacker(self):
        cmp = _make_computer(def_pylon=7.0)
        assert abs(cmp.episode_terminal_bonus(_sq(role="attack"), _gs())) < 1e-9


class TestRewardConfigValidation:
    def test_step_penalty_must_be_non_positive(self):
        with pytest.raises(Exception):
            AttackerRewardConfig(step_penalty=0.01)

    def test_all_bonuses_non_negative(self):
        AttackerRewardConfig(
            step_penalty=-0.001,
            pylon_destroy_bonus=5.0,
            unit_kill_bonus=1.0,
            target_type_bonus=2.0,
            target_type_bonus_clip=5.0,
        )
