"""Macro composition / build-order contract penalties for the manager agent."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from sc2.ids.unit_typeid import UnitTypeId

from src.agent.commands import Command, CommandType

# Buildings that are typically unique early (surplus = waste).
_DEFAULT_MAX_COUNTS = {
    UnitTypeId.SPAWNINGPOOL: 1,
    UnitTypeId.ROACHWARREN: 1,
    UnitTypeId.HYDRALISKDEN: 1,
}


def evaluate_manager_emission(
    command: Command,
    *,
    has_pool: bool,
    structure_counts: Dict[UnitTypeId, int],
    required_max: Optional[Dict[UnitTypeId, int]] = None,
) -> Optional[str]:
    """
    Return a penalty key if the emission violates curriculum BO / surplus rules.

    Keys: ``wrong_build_order``, ``surplus_building``, ``invalid_command``.
    """
    caps = required_max or _DEFAULT_MAX_COUNTS

    if command.command_type == CommandType.TRAIN:
        ut = command.unit_type_id
        if ut in {UnitTypeId.ZERGLING, UnitTypeId.ZERGLINGBURROWED} and not has_pool:
            return "wrong_build_order"
        if ut in {UnitTypeId.ROACH, UnitTypeId.ROACHBURROWED}:
            warren = structure_counts.get(UnitTypeId.ROACHWARREN, 0)
            if warren < 1:
                return "wrong_build_order"
        if ut == UnitTypeId.QUEEN and not has_pool:
            return "wrong_build_order"
        return None

    if command.command_type == CommandType.BUILD and command.unit_type_id is not None:
        ut = command.unit_type_id
        max_allowed = caps.get(ut)
        if max_allowed is not None and structure_counts.get(ut, 0) >= max_allowed:
            return "surplus_building"
        return None

    if command.command_type == CommandType.INJECT:
        # Energy / queen presence is an invalid-command case at emit time if known.
        return None

    return None


def penalty_amount(rewards_cfg: Dict[str, Any], key: str) -> float:
    mapping = {
        "wrong_build_order": "wrong_build_order_penalty",
        "surplus_building": "surplus_building_penalty",
        "invalid_command": "invalid_command_penalty",
    }
    reward_key = mapping.get(key, key)
    return float(rewards_cfg.get(reward_key, -5.0))
