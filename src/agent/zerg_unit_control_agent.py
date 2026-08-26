"""Specialised Zerg unit-control agent (universal architecture, Zerg-only training).

Stub for the future frozen multi-agent curriculum: manager / build / Zerg UC train
one-at-a-time with the others frozen. Opponent seating with a UC checkpoint is
wired later via ``env.opponent.mode: unit_control``.
"""

from src.agent.universal_unit_control_agent import UniversalUnitControlAgent


class ZergUnitControlAgent(UniversalUnitControlAgent):
    """Universal unit-control contract, specialised for Zerg-only curricula."""

    agent_kind = "zerg_unit_control"
