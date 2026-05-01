"""
WinConditionChecker — stateless episode-termination logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from train.envs.ctf.squad_manager import Outcome, SquadState, SquadStatus


@dataclass
class GameState:
    step: int
    max_steps: int
    learner_pylons: int
    learner_units_alive: int
    enemy_pylons_alive: int
    enemy_units_alive: int
    learner_squads: List[SquadState] = field(default_factory=list)


@dataclass
class TerminalResult:
    done: bool
    outcome: Outcome
    reason: str = ""


class WinConditionChecker:
    def check(self, gs: GameState) -> TerminalResult:
        if gs.learner_pylons == 0 or gs.learner_units_alive == 0:
            return TerminalResult(done=True, outcome=Outcome.LOSS, reason="learner_wiped")

        if gs.enemy_units_alive == 0:
            return TerminalResult(done=True, outcome=Outcome.WIN, reason="all_enemy_dead")

        active_roles = {
            sq.role
            for sq in gs.learner_squads
            if sq.status == SquadStatus.ACTIVE
        }

        if active_roles == {"attack"}:
            if gs.enemy_pylons_alive == 0:
                return TerminalResult(done=True, outcome=Outcome.WIN, reason="all_enemy_pylons_dead")

        elif active_roles == {"defend"}:
            if gs.enemy_units_alive == 0:
                return TerminalResult(done=True, outcome=Outcome.WIN, reason="all_enemy_dead_defend")
            if gs.step >= gs.max_steps:
                survived = gs.learner_pylons > 0 and gs.learner_units_alive > 0
                outcome = Outcome.WIN if survived else Outcome.LOSS
                return TerminalResult(done=True, outcome=outcome, reason="timeout_defend")

        elif active_roles == {"attack", "defend"} or (
            "attack" in active_roles and "defend" in active_roles
        ):
            attack_done = self._all_attackers_done(gs)
            defend_done = self._all_defenders_done(gs)
            if attack_done and defend_done:
                return TerminalResult(done=True, outcome=Outcome.WIN, reason="mixed_objectives_met")

        if gs.step >= gs.max_steps:
            return TerminalResult(done=True, outcome=Outcome.LOSS, reason="timeout_non_defend")

        return TerminalResult(done=False, outcome=Outcome.ONGOING)

    def _all_attackers_done(self, gs: GameState) -> bool:
        active_attackers = [
            sq for sq in gs.learner_squads
            if sq.role == "attack" and sq.status == SquadStatus.ACTIVE
        ]
        if not active_attackers:
            return True
        return all(sq.assigned_pylon_destroyed for sq in active_attackers)

    def _all_defenders_done(self, gs: GameState) -> bool:
        active_defenders = [
            sq for sq in gs.learner_squads
            if sq.role == "defend" and sq.status == SquadStatus.ACTIVE
        ]
        if not active_defenders:
            return True
        for sq in active_defenders:
            tags = sq.assigned_attacker_unit_tags
            if len(tags) == 0:
                continue
            if not _attacker_tags_all_dead(sq):
                return False
        return True


def _attacker_tags_all_dead(sq: SquadState) -> bool:
    alive_count = getattr(sq, "_alive_attacker_tags_count", None)
    if alive_count is None:
        return len(sq.assigned_attacker_unit_tags) == 0
    return alive_count == 0
