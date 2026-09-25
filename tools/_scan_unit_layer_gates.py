"""Find CTF/scenario/AoE levels that spawn units before matching map layers exist."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "config_universal"
PIPELINE = yaml.safe_load((ROOT / "pipeline.yaml").read_text())["levels"]

GATES = {
    "flying": [
        "allied_unit_flying",
        "enemy_unit_flying",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ],
    "massive": ["allied_unit_massive", "enemy_unit_massive"],
    "armored": ["allied_unit_armored", "enemy_unit_armored"],
    "bio_mech_light": [
        "allied_unit_biological",
        "enemy_unit_biological",
        "allied_unit_mechanical",
        "enemy_unit_mechanical",
        "allied_unit_light",
        "enemy_unit_light",
    ],
    "splash": ["allied_splash", "enemy_splash"],
    "cloak": ["allied_unit_cloak", "enemy_unit_cloak"],
}

# Air / transport units that need flying map channels.
FLYING = {
    "BATTLECRUISER",
    "CORRUPTOR",
    "MUTALISK",
    "VIPER",
    "BROODLORD",
    "OVERLORD",
    "OVERLORDTRANSPORT",
    "OVERSEER",
    "BANSHEE",
    "VIKINGFIGHTER",
    "VIKINGASSAULT",
    "LIBERATOR",
    "LIBERATORAG",
    "MEDIVAC",
    "RAVEN",
    "ORACLE",
    "PHOENIX",
    "VOIDRAY",
    "CARRIER",
    "TEMPEST",
    "MOTHERSHIP",
    "OBSERVER",
    "WARPPRISM",
}

# Massive units that need massive overlay channels.
MASSIVE = {
    "THOR",
    "THORAP",
    "ULTRALISK",
    "COLOSSUS",
    "ARCHON",
    "CARRIER",
    "TEMPEST",
    "MOTHERSHIP",
    "BATTLECRUISER",
    "BROODLORD",
}


def _layer_intro() -> dict[str, str]:
    intro: dict[str, str] = {}
    seen: set[str] = set()
    for name in PIPELINE:
        path = ROOT / f"{name}.yaml"
        if not path.exists():
            continue
        env = (yaml.safe_load(path.read_text()) or {}).get("env") or {}
        for key in ("active_layers", "active_tactical_layers"):
            for layer in env.get(key) or []:
                if layer not in seen:
                    seen.add(layer)
                    intro[layer] = name
    return intro


def _gate_ready(level: str, gate: str, intro: dict[str, str]) -> bool:
    idx = PIPELINE.index(level)
    return all(
        intro.get(layer) is not None and PIPELINE.index(intro[layer]) <= idx
        for layer in GATES[gate]
    )


def _units(level: str) -> list[str]:
    path = ROOT / f"{level}.yaml"
    if not path.exists():
        return []
    env = (yaml.safe_load(path.read_text()) or {}).get("env") or {}
    out: list[str] = []
    for pool in env.get("unit_pools") or []:
        out.extend(pool.get("unit_pool") or [])
    return sorted(set(out))


def main() -> None:
    intro = _layer_intro()
    print("Gate intros:")
    for gate, layers in GATES.items():
        print(f"  {gate}: {intro.get(layers[0])}")

    print("\nViolations:")
    for name in PIPELINE:
        if not name.startswith(("ctf_", "scenario_", "aoe_")):
            continue
        units = _units(name)
        if not units:
            continue
        problems: list[tuple[str, list[str]]] = []
        flyers = sorted(u for u in units if u in FLYING)
        if flyers and not _gate_ready(name, "flying", intro):
            problems.append(("flying", flyers))
        massive = sorted(u for u in units if u in MASSIVE)
        if massive and not _gate_ready(name, "massive", intro):
            problems.append(("massive", massive))
        if problems:
            print(name)
            for gate, us in problems:
                print(f"  {gate}: {us}")


if __name__ == "__main__":
    main()
