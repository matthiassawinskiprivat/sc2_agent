"""
RoleRewardComputer — per-step reward computation for CTF squads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from train.envs.ctf.config import RewardConfig
from train.envs.ctf.game_data import get_weapon_bonus
from train.envs.ctf.squad_manager import SquadState
from train.envs.ctf.win_condition import GameState


@dataclass
class WeaponFireRewardEvent:
    attacker_tag: int
    attacker_type_id: int
    target_tag: int
    target_type_id: int


@dataclass
class UnitKillRewardEvent:
    victim_tag: int
    victim_type_id: int


@dataclass
class PylonDestroyRewardEvent:
    pylon_tag: int


@dataclass
class StepEvents:
    weapon_fires: List[WeaponFireRewardEvent] = field(default_factory=list)
    unit_kills: List[UnitKillRewardEvent] = field(default_factory=list)
    pylon_destroy: List[PylonDestroyRewardEvent] = field(default_factory=list)

    def absorb(self, raw_event: Any) -> None:
        event_type = type(raw_event).__name__
        if event_type == "WeaponFireEvent":
            self.weapon_fires.append(WeaponFireRewardEvent(
                attacker_tag=raw_event.attacker_tag,
                attacker_type_id=getattr(raw_event, "attacker_type_id", 0),
                target_tag=raw_event.target_tag,
                target_type_id=raw_event.target_type_id,
            ))
        elif event_type == "UnitDeathEvent":
            self.unit_kills.append(UnitKillRewardEvent(
                victim_tag=raw_event.tag,
                victim_type_id=getattr(raw_event, "type_id", 0),
            ))
        elif event_type == "PylonDeathEvent":
            self.pylon_destroy.append(PylonDestroyRewardEvent(pylon_tag=raw_event.tag))


class RoleRewardComputer:
    def __init__(self, reward_config: RewardConfig) -> None:
        self._cfg = reward_config

    def compute(self, squad: SquadState, events: StepEvents, gs: GameState) -> float:
        if squad.role == "attack":
            return self._attacker_reward(squad, events, gs)
        return self._defender_reward(squad, events, gs)

    def _attacker_reward(self, squad: SquadState, events: StepEvents, gs: GameState) -> float:
        cfg = self._cfg.attacker
        total = cfg.step_penalty
        if events.pylon_destroy:
            total += cfg.pylon_destroy_bonus * len(events.pylon_destroy)
        total += cfg.unit_kill_bonus * len(events.unit_kills)
        ttb = self._compute_target_type_bonus(events, cfg.target_type_bonus)
        total += min(ttb, cfg.target_type_bonus_clip)
        return total

    def _defender_reward(self, squad: SquadState, events: StepEvents, gs: GameState) -> float:
        cfg = self._cfg.defender
        total = cfg.step_reward
        total += cfg.unit_kill_bonus * len(events.unit_kills)
        ttb = self._compute_target_type_bonus(events, cfg.target_type_bonus)
        total += min(ttb, cfg.target_type_bonus_clip)
        return total

    def episode_terminal_bonus(self, squad: SquadState, gs: GameState) -> float:
        if squad.role != "defend":
            return 0.0
        if not squad.assigned_pylon_destroyed:
            return self._cfg.defender.pylon_survival_bonus
        return 0.0

    def _compute_target_type_bonus(self, events: StepEvents, bonus_scalar: float) -> float:
        total = 0.0
        for ev in events.weapon_fires:
            raw_bonus = get_weapon_bonus(ev.attacker_type_id, ev.target_type_id)
            if raw_bonus > 0.0:
                total += raw_bonus * bonus_scalar
        return total
