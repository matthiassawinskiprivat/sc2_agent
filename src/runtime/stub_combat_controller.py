"""Heuristic combat for manager ATTACK/DEFEND (no unit-control PPO)."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import Command, CommandType

# Workers / larva / eggs are never part of stub army orders.
_NON_ARMY = {
    UnitTypeId.DRONE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.LARVA,
    UnitTypeId.EGG,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERSEER,
    UnitTypeId.QUEEN,
}


def _is_army_unit(unit: Any) -> bool:
    type_id = getattr(unit, "type_id", None)
    if type_id in _NON_ARMY:
        return False
    if bool(getattr(unit, "is_structure", False)):
        return False
    if bool(getattr(unit, "is_worker", False)):
        return False
    return True


class StubCombatController:
    """Apply standing ATTACK (attack-move) / DEFEND (move) to non-worker army."""

    def apply(self, bot: Any, combat_cmd: Optional[Command]) -> None:
        if combat_cmd is None:
            return
        target = combat_cmd.target_pos
        if target is None:
            hatch = getattr(bot, "townhalls", None)
            if hatch and getattr(hatch, "exists", False):
                target = hatch.first.position
            else:
                return
        if not isinstance(target, Point2):
            target = Point2((float(target[0]), float(target[1])))

        army = [u for u in self._iter_own_units(bot) if _is_army_unit(u)]
        if not army:
            return

        do = getattr(bot, "do", None)
        if do is None:
            return

        if combat_cmd.command_type == CommandType.ATTACK:
            for unit in army:
                do(unit.attack(target))
        elif combat_cmd.command_type == CommandType.DEFEND:
            for unit in army:
                do(unit.move(target))

    @staticmethod
    def _iter_own_units(bot: Any) -> Iterable[Any]:
        units = getattr(bot, "units", None)
        if units is None:
            return []
        if hasattr(units, "__iter__"):
            return list(units)
        return []
