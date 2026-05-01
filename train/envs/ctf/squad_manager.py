"""
SquadManager — squad state, round-robin iteration, and reassignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from train.envs.ctf.config import CTFConfig, SquadConfig, _ROLE, _SIDE
from train.envs.ctf.region_loader import Point2, RegionDef


class SquadStatus(Enum):
    ACTIVE = auto()
    GHOST = auto()
    ELIMINATED = auto()


class Outcome(Enum):
    ONGOING = auto()
    WIN = auto()
    LOSS = auto()


@dataclass
class UnitDeathEvent:
    tag: int
    position: Point2


@dataclass
class PylonDeathEvent:
    tag: int
    position: Point2


@dataclass
class WeaponFireEvent:
    attacker_tag: int
    target_tag: int
    target_type_id: int


@dataclass
class GameEvent:
    unit_deaths: List[UnitDeathEvent] = field(default_factory=list)
    pylon_deaths: List[PylonDeathEvent] = field(default_factory=list)
    weapon_fires: List[WeaponFireEvent] = field(default_factory=list)


@dataclass
class SquadState:
    squad_id: str
    role: _ROLE
    side: _SIDE
    status: SquadStatus
    unit_tags: Set[int] = field(default_factory=set)

    current_region: Optional[str] = None
    objective_pylon_tag: Optional[int] = None
    assigned_pylon_destroyed: bool = False
    assigned_attacker_unit_tags: FrozenSet[int] = field(default_factory=frozenset)

    @property
    def is_active(self) -> bool:
        return self.status == SquadStatus.ACTIVE

    @property
    def unit_count(self) -> int:
        return len(self.unit_tags)

    @property
    def assigned_attacker_units_dead(self) -> bool:
        return len(self.assigned_attacker_unit_tags) > 0 and len(
            self.assigned_attacker_unit_tags
        ) == 0

    def transfer_units_to(self, target: "SquadState") -> None:
        target.unit_tags.update(self.unit_tags)
        self.unit_tags.clear()
        self.status = SquadStatus.GHOST


def compare_id(a: str, b: str) -> int:
    """
    Compare two squad ids for tie-breaking.

    Returns -1 if a < b, 0 if equal, +1 if a > b.
    Purely-numeric ids are compared as integers; otherwise lexicographic.
    """
    def key(s: str):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    ka, kb = key(a), key(b)
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


class SquadManager:
    def __init__(
        self,
        squad_configs: List[SquadConfig],
        learner_side: _SIDE,
        regions: Dict[str, RegionDef],
    ) -> None:
        self._learner_side = learner_side
        self._opponent_side: _SIDE = "player_b" if learner_side == "player_a" else "player_a"
        self._regions = regions

        self._states: Dict[str, SquadState] = {}
        for sq_cfg in squad_configs:
            initial_region = (
                sq_cfg.attack_targets[0] if sq_cfg.role == "attack" and sq_cfg.attack_targets
                else sq_cfg.defend_anchors[0] if sq_cfg.role == "defend" and sq_cfg.defend_anchors
                else sq_cfg.spawn_region
            )
            self._states[sq_cfg.id] = SquadState(
                squad_id=sq_cfg.id,
                role=sq_cfg.role,
                side=sq_cfg.side,
                status=SquadStatus.ACTIVE,
                current_region=initial_region,
            )

        self._learner_ids: List[str] = [
            sq.id for sq in squad_configs if sq.side == learner_side
        ]
        self._opponent_ids: List[str] = [
            sq.id for sq in squad_configs if sq.side != learner_side
        ]

        self._learner_rr_idx: int = 0
        self._opponent_rr_idx: int = 0
        self._tag_to_squad: Dict[int, str] = {}
        self._squad_cfg_map: Dict[str, SquadConfig] = {sq.id: sq for sq in squad_configs}
        self._alive_attacker_tags: Set[int] = set()

    def bind_objectives(self, objective_resolver: Any) -> None:
        pylon_tags = objective_resolver.all_pylon_tags()
        for state in self._states.values():
            cfg = self._squad_cfg_map[state.squad_id]
            if state.role == "attack" and cfg.attack_targets:
                for tgt in cfg.attack_targets:
                    if tgt in pylon_tags:
                        state.objective_pylon_tag = pylon_tags[tgt]
                        break
            elif state.role == "defend" and cfg.defend_anchors:
                for anc in cfg.defend_anchors:
                    if anc in pylon_tags:
                        state.objective_pylon_tag = pylon_tags[anc]
                        break

    def register_unit_tags(self, squad_id: str, tags: List[int]) -> None:
        state = self._states[squad_id]
        state.unit_tags.update(tags)
        for tag in tags:
            self._tag_to_squad[tag] = squad_id

    def advance_learner_rr(self) -> Optional[SquadState]:
        return self._advance_rr(self._learner_ids, "_learner_rr_idx")

    def advance_opponent_rr(self) -> Optional[SquadState]:
        return self._advance_rr(self._opponent_ids, "_opponent_rr_idx")

    def current_learner_squad(self) -> Optional[SquadState]:
        return self._peek_rr(self._learner_ids, self._learner_rr_idx)

    def current_opponent_squad(self) -> Optional[SquadState]:
        return self._peek_rr(self._opponent_ids, self._opponent_rr_idx)

    def process_events(self, events: GameEvent) -> None:
        for ev in events.unit_deaths:
            squad_id = self._tag_to_squad.pop(ev.tag, None)
            if squad_id:
                state = self._states[squad_id]
                state.unit_tags.discard(ev.tag)
                if state.unit_count == 0 and state.status == SquadStatus.ACTIVE:
                    state.status = SquadStatus.ELIMINATED
            self._alive_attacker_tags.discard(ev.tag)

        for ev in events.pylon_deaths:
            for state in self._states.values():
                if state.objective_pylon_tag == ev.tag:
                    if state.role == "attack":
                        state.assigned_pylon_destroyed = True

    def check_and_reassign(self, regions: Dict[str, RegionDef]) -> None:
        self._regions = regions

        for sq_id in list(self._learner_ids + self._opponent_ids):
            state = self._states[sq_id]
            if (
                state.status == SquadStatus.ACTIVE
                and state.role == "attack"
                and state.assigned_pylon_destroyed
                and state.unit_count > 0
            ):
                self._reassign_attacker_done(state)

        for sq_id in list(self._learner_ids + self._opponent_ids):
            state = self._states[sq_id]
            if state.status == SquadStatus.ELIMINATED and state.role == "attack":
                paired_def = self._find_paired_defender(sq_id)
                if paired_def is not None and paired_def.status == SquadStatus.ACTIVE:
                    self._reassign_eliminated_attacker_defender(paired_def)
                    state.status = SquadStatus.GHOST

    def unit_to_squad(self, tag: Optional[int]) -> Optional[str]:
        if tag is None:
            return None
        return self._tag_to_squad.get(tag)

    def is_learner_squad(self, squad_id: str) -> bool:
        return self._states[squad_id].side == self._learner_side

    def learner_squad_states(self) -> List[SquadState]:
        return [self._states[sid] for sid in self._learner_ids]

    def opponent_squad_states(self) -> List[SquadState]:
        return [self._states[sid] for sid in self._opponent_ids]

    def active_learner_squads(self) -> List[SquadState]:
        return [s for s in self.learner_squad_states() if s.is_active]

    def active_opponent_squads(self) -> List[SquadState]:
        return [s for s in self.opponent_squad_states() if s.is_active]

    def learner_pylon_count(self) -> int:
        alive = 0
        for state in self.learner_squad_states():
            if state.role == "defend" and not state.assigned_pylon_destroyed:
                alive += 1
        return alive

    def learner_unit_count(self) -> int:
        return sum(s.unit_count for s in self.learner_squad_states())

    def opponent_pylon_count(self) -> int:
        alive = 0
        for state in self.opponent_squad_states():
            if state.role == "defend" and not state.assigned_pylon_destroyed:
                alive += 1
        return alive

    def opponent_unit_count(self) -> int:
        return sum(s.unit_count for s in self.opponent_squad_states())

    def get_state(self, squad_id: str) -> SquadState:
        return self._states[squad_id]

    def _reassign_attacker_done(self, source: SquadState) -> None:
        allies = [
            s for s in self._all_active_states()
            if s.side == source.side and s.squad_id != source.squad_id
        ]
        if not allies:
            source.status = SquadStatus.GHOST
            return
        target = self._closest_ally(source.current_region, allies)
        source.transfer_units_to(target)
        for tag in target.unit_tags:
            self._tag_to_squad[tag] = target.squad_id

    def _reassign_eliminated_attacker_defender(self, defender: SquadState) -> None:
        allies = [
            s for s in self._all_active_states()
            if s.side == defender.side and s.squad_id != defender.squad_id
        ]
        if not allies:
            defender.status = SquadStatus.GHOST
            return
        target = self._closest_ally(defender.current_region, allies)
        defender.transfer_units_to(target)
        for tag in target.unit_tags:
            self._tag_to_squad[tag] = target.squad_id

    def _closest_ally(
        self,
        source_region: Optional[str],
        candidates: List[SquadState],
    ) -> SquadState:
        source_center = self._region_center(source_region)
        best: Optional[SquadState] = None
        best_dist: float = math.inf
        for cand in candidates:
            cand_center = self._region_center(cand.current_region)
            dist = source_center.distance_to(cand_center)
            if dist < best_dist:
                best_dist = dist
                best = cand
            elif dist == best_dist and best is not None:
                if compare_id(cand.squad_id, best.squad_id) < 0:
                    best = cand
        assert best is not None
        return best

    def _find_paired_defender(self, attacker_squad_id: str) -> Optional[SquadState]:
        """
        Find the defender (any side) whose defend_anchors share a region
        name with the eliminated attacker's attack_targets.

        In a standard CTF layout the pairing is cross-side: player_a attacks
        obj_b, which player_b defends.  When the attacker is eliminated the
        opposing defender at that objective no longer has a threat to guard
        and is freed to reinforce.
        """
        atk_cfg = self._squad_cfg_map[attacker_squad_id]
        atk_targets = set(atk_cfg.attack_targets)
        for sq_id, state in self._states.items():
            if state.role != "defend":
                continue
            def_cfg = self._squad_cfg_map[sq_id]
            if atk_targets & set(def_cfg.defend_anchors):
                return state
        return None

    def _region_center(self, region_name: Optional[str]) -> Point2:
        if region_name and region_name in self._regions:
            return self._regions[region_name].center
        return Point2(0.0, 0.0)

    def _all_active_states(self) -> List[SquadState]:
        return [s for s in self._states.values() if s.status == SquadStatus.ACTIVE]

    def _advance_rr(self, id_list: List[str], idx_attr: str) -> Optional[SquadState]:
        if not id_list:
            return None
        n = len(id_list)
        start = getattr(self, idx_attr)
        for offset in range(n):
            idx = (start + offset) % n
            state = self._states[id_list[idx]]
            if state.status == SquadStatus.ACTIVE:
                setattr(self, idx_attr, (idx + 1) % n)
                return state
        return None

    def _peek_rr(self, id_list: List[str], idx: int) -> Optional[SquadState]:
        if not id_list:
            return None
        state = self._states[id_list[idx % len(id_list)]]
        return state if state.status == SquadStatus.ACTIVE else None
