"""Scan CTF curriculum for missing command-enabling units after each head intro."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "config_universal"
PIPELINE = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))["levels"]

# Gate levels (first pipeline index where the feature is legal).
GATES = {
    "heal": "ctf_heal_queen_vs_queen",
    "cloak": "ctf_level_21",
    "aoe": "aoe_high_templar_zealot_vs_marine_marauder",
    "snipe": "snipe_ghost_marauder_vs_roach_hydra",
    "flying": "scenario_queen_vs_battlecruiser",
    "drop": "ctf_level_24",
}

HEALERS = {
    "zerg": ("QUEEN",),
    "terran": None,  # composition-dependent
}
BIO = {"MARINE", "MARAUDER", "REAPER", "GHOST"}
MECH = {
    "HELLION",
    "HELLIONTANK",
    "CYCLONE",
    "SIEGETANK",
    "THOR",
    "THORAP",
    "WIDOWMINE",
    "BANSHEE",
    "VIKINGFIGHTER",
    "VIKINGASSAULT",
    "LIBERATOR",
    "LIBERATORAG",
    "BATTLECRUISER",
    "RAVEN",
}

# Race-appropriate AoE casters (after aoe gate).
AOE_BY_RACE = {
    "zerg": ("INFESTOR", "RAVAGER"),  # VIPER needs air gate + air enemy
    "terran": ("GHOST",),
    "protoss": ("HIGHTEMPLAR",),
}

# Race-appropriate snipe casters (after snipe gate).
SNIPE_BY_RACE = {
    "zerg": ("INFESTOR",),
    "terran": ("GHOST",),  # BC needs flying gate
    "protoss": ("HIGHTEMPLAR",),
}

# After flying + aoe: VIPER on zerg pools that face air or are air themselves.
AIR_FOR_VIPER = {
    "BATTLECRUISER",
    "CORRUPTOR",
    "MUTALISK",
    "VIPER",
    "BROODLORD",
    "BANSHEE",
    "VIKINGFIGHTER",
    "VIKINGASSAULT",
    "LIBERATOR",
    "LIBERATORAG",
    "RAVEN",
    "ORACLE",
    "PHOENIX",
    "VOIDRAY",
    "CARRIER",
    "TEMPEST",
    "MOTHERSHIP",
}


def _load_pools(name: str) -> list[dict]:
    cfg = yaml.safe_load((ROOT / f"{name}.yaml").read_text(encoding="utf-8"))
    return list((cfg.get("env") or {}).get("unit_pools") or [])


def _suggest_healers(race: str, units: set[str], *, flying_ready: bool) -> list[str]:
    if race == "zerg":
        return [] if "QUEEN" in units else ["QUEEN"]
    if race == "terran":
        out: list[str] = []
        if units & BIO and flying_ready and "MEDIVAC" not in units:
            out.append("MEDIVAC")
        if units & MECH and "SCV" not in units:
            out.append("SCV")
        return out
    return []


def _suggest_aoe(race: str, units: set[str], *, aoe_ready: bool, flying_ready: bool) -> list[str]:
    if not aoe_ready:
        return []
    out: list[str] = []
    if race == "zerg":
        if "INFESTOR" not in units:
            out.append("INFESTOR")
        if "RAVAGER" not in units and ("ROACH" in units or "RAVAGER" in units):
            # Prefer Ravager when ground zerg composition exists.
            if units & {"ROACH", "ZERGLING", "HYDRALISK", "RAVAGER"}:
                out.append("RAVAGER")
        if flying_ready and "VIPER" not in units:
            # Viper only when this squad or the matchup already has air context
            # (caller filters by enemy/ally air).
            pass  # handled separately
    elif race == "terran":
        if "GHOST" not in units:
            out.append("GHOST")
    elif race == "protoss":
        if "HIGHTEMPLAR" not in units:
            out.append("HIGHTEMPLAR")
    return out


def _suggest_snipe(race: str, units: set[str], *, snipe_ready: bool, flying_ready: bool) -> list[str]:
    if not snipe_ready:
        return []
    out: list[str] = []
    if race == "zerg" and "INFESTOR" not in units:
        out.append("INFESTOR")
    elif race == "terran":
        if "GHOST" not in units:
            out.append("GHOST")
        if flying_ready and "BATTLECRUISER" not in units and units & AIR_FOR_VIPER:
            # Only suggest BC when the pool already has air (don't force BC everywhere).
            pass
    elif race == "protoss" and "HIGHTEMPLAR" not in units:
        out.append("HIGHTEMPLAR")
    return out


def main() -> None:
    gate_i = {k: PIPELINE.index(v) for k, v in GATES.items()}
    key_counts: Counter[str] = Counter()
    presence: dict[str, list[str]] = defaultdict(list)

    heal_adds: dict[str, list[tuple]] = defaultdict(list)
    aoe_adds: dict[str, list[tuple]] = defaultdict(list)
    snipe_adds: dict[str, list[tuple]] = defaultdict(list)
    viper_adds: dict[str, list[tuple]] = defaultdict(list)

    for i, name in enumerate(PIPELINE):
        if not name.startswith(("ctf_", "scenario_", "aoe_", "snipe_")):
            continue
        pools = _load_pools(name)
        all_units: set[str] = set()
        for p in pools:
            all_units |= set(p.get("unit_pool") or [])
        for u in all_units:
            key_counts[u] += 1
            presence[u].append(name)

        heal_ready = i >= gate_i["heal"]
        aoe_ready = i >= gate_i["aoe"]
        snipe_ready = i >= gate_i["snipe"]
        flying_ready = i >= gate_i["flying"]

        # Per-matchup air: any pool has air → Viper useful on zerg side.
        matchup_has_air = bool(all_units & AIR_FOR_VIPER)

        for p in pools:
            race = p["race"]
            units = set(p.get("unit_pool") or [])
            role = p["high_level_command"]
            region = p["region"]
            if heal_ready:
                for u in _suggest_healers(race, units, flying_ready=flying_ready):
                    heal_adds[name].append((race, role, region, u))
            if aoe_ready:
                for u in _suggest_aoe(race, units, aoe_ready=True, flying_ready=flying_ready):
                    aoe_adds[name].append((race, role, region, u))
            if snipe_ready:
                for u in _suggest_snipe(
                    race, units, snipe_ready=True, flying_ready=flying_ready
                ):
                    snipe_adds[name].append((race, role, region, u))
            if (
                aoe_ready
                and flying_ready
                and race == "zerg"
                and matchup_has_air
                and "VIPER" not in units
            ):
                viper_adds[name].append((race, role, region, "VIPER"))

    print("=== Key unit presence ===")
    for u in (
        "QUEEN",
        "MEDIVAC",
        "SCV",
        "GHOST",
        "INFESTOR",
        "HIGHTEMPLAR",
        "RAVAGER",
        "VIPER",
        "BATTLECRUISER",
    ):
        print(f"  {u}: {key_counts[u]} levels -> {presence[u]}")

    def _summary(title: str, data: dict[str, list[tuple]]) -> None:
        print(f"\n=== {title}: {len(data)} levels ===")
        for name in PIPELINE:
            if name not in data:
                continue
            units = sorted({t[3] for t in data[name]})
            print(f"  {name}: +{units} ({len(data[name])} pool slots)")

    _summary("HEAL adds", heal_adds)
    _summary("AOE caster adds (Infestor/Ravager/Ghost/HT)", aoe_adds)
    _summary("SNIPE caster adds", snipe_adds)
    _summary("VIPER adds (after flying+aoe, air matchup)", viper_adds)


if __name__ == "__main__":
    main()
