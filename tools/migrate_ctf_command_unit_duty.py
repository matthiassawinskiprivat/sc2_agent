"""Add command-enabling units to CTF pools after each head / layer gate.

Rules (pipeline order):
  heal head (@ ctf_heal_queen_vs_queen)
    - Zerg pools: QUEEN
    - Terran pools with mech: SCV
    - Terran pools with bio: MEDIVAC (only after flying gate)
  aoe / snipe heads
    - Zerg: INFESTOR if missing; RAVAGER if ROACH present
    - Terran: GHOST if missing
    - Protoss: HIGHTEMPLAR if missing
  flying + aoe (@ scenario_queen_vs_battlecruiser)
    - Zerg pools in air matchups: VIPER

Protoss never gets a healer. Medivac/Viper never appear before the flying gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "config_universal"
PIPELINE = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))["levels"]

GATES = {
    "heal": "ctf_heal_queen_vs_queen",
    "aoe": "aoe_high_templar_zealot_vs_marine_marauder",
    "snipe": "snipe_ghost_marauder_vs_roach_hydra",
    "flying": "scenario_queen_vs_battlecruiser",
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
AIR = {
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
    "OVERLORDTRANSPORT",
}
# Parasitic Bomb needs combat air — not support/transports we add for heal/drop.
AIR_FOR_VIPER = AIR - {
    "MEDIVAC",
    "WARPPRISM",
    "OVERLORDTRANSPORT",
    "OBSERVER",
    "OVERLORD",
    "OVERSEER",
}

_UNIT_LINE = re.compile(r"^(\s+)-\s+([A-Z][A-Z0-9_]*)\s*$")


def _desired_adds(
    *,
    race: str,
    units: set[str],
    matchup_air: bool,
    heal_ready: bool,
    aoe_ready: bool,
    snipe_ready: bool,
    flying_ready: bool,
) -> list[str]:
    add: list[str] = []

    def _push(name: str) -> None:
        if name not in units and name not in add:
            add.append(name)

    if heal_ready:
        if race == "zerg":
            _push("QUEEN")
        elif race == "terran":
            if units & MECH:
                _push("SCV")
            if flying_ready and units & BIO:
                _push("MEDIVAC")

    if aoe_ready or snipe_ready:
        if race == "zerg":
            _push("INFESTOR")
            if "ROACH" in units:
                _push("RAVAGER")
        elif race == "terran":
            _push("GHOST")
        elif race == "protoss":
            _push("HIGHTEMPLAR")

    if aoe_ready and flying_ready and race == "zerg" and matchup_air:
        _push("VIPER")

    return add


def _append_units_by_pool_index(
    text: str, adds_by_index: dict[int, list[str]]
) -> tuple[str, list[str]]:
    """Append units when leaving each ``unit_pool`` list (pools are 0-indexed)."""
    if not adds_by_index:
        return text, []

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    applied: list[str] = []
    pool_i = -1
    in_unit_pool = False
    unit_indent = "    - "
    pending_adds: list[str] | None = None

    def _flush() -> None:
        nonlocal pending_adds, in_unit_pool
        if pending_adds:
            for u in pending_adds:
                out.append(f"{unit_indent}{u}\n")
                applied.append(u)
        pending_adds = None
        in_unit_pool = False

    for line in lines:
        stripped = line.lstrip(" ")
        if stripped.startswith("unit_pool:"):
            if in_unit_pool:
                _flush()
            pool_i += 1
            in_unit_pool = True
            pending_adds = list(adds_by_index.get(pool_i) or [])
            indent = line[: len(line) - len(stripped)]
            unit_indent = f"{indent}  - "
            out.append(line)
            continue

        if in_unit_pool:
            m = _UNIT_LINE.match(line.rstrip("\n\r"))
            if m:
                unit_indent = f"{m.group(1)}- "
                out.append(line)
                continue
            _flush()
            out.append(line)
            continue

        out.append(line)

    if in_unit_pool:
        _flush()

    return "".join(out), applied


def main() -> int:
    gate_i = {k: PIPELINE.index(v) for k, v in GATES.items()}
    changed_files = 0
    total_adds = 0

    for i, name in enumerate(PIPELINE):
        if not name.startswith(("ctf_", "scenario_", "aoe_", "snipe_")):
            continue
        path = ROOT / f"{name}.yaml"
        if not path.exists():
            continue

        heal_ready = i >= gate_i["heal"]
        aoe_ready = i >= gate_i["aoe"]
        snipe_ready = i >= gate_i["snipe"]
        flying_ready = i >= gate_i["flying"]
        if not (heal_ready or aoe_ready or snipe_ready):
            continue

        raw = path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw) or {}
        pools = list((cfg.get("env") or {}).get("unit_pools") or [])
        if not pools:
            continue

        all_units: set[str] = set()
        for p in pools:
            all_units |= set(p.get("unit_pool") or [])
        matchup_air = bool(all_units & AIR_FOR_VIPER)

        adds_by_index: dict[int, list[str]] = {}
        for pi, p in enumerate(pools):
            add = _desired_adds(
                race=p["race"],
                units=set(p.get("unit_pool") or []),
                matchup_air=matchup_air,
                heal_ready=heal_ready,
                aoe_ready=aoe_ready,
                snipe_ready=snipe_ready,
                flying_ready=flying_ready,
            )
            if add:
                adds_by_index[pi] = add

        if not adds_by_index:
            continue

        new_text, applied = _append_units_by_pool_index(raw, adds_by_index)
        if new_text == raw or not applied:
            print(f"{name}: no text change")
            continue

        # Verify parse still matches expected pool sizes.
        new_cfg = yaml.safe_load(new_text) or {}
        new_pools = list((new_cfg.get("env") or {}).get("unit_pools") or [])
        if len(new_pools) != len(pools):
            raise RuntimeError(f"{name}: pool count changed")
        for pi, (old_p, new_p) in enumerate(zip(pools, new_pools)):
            expected = set(old_p.get("unit_pool") or []) | set(adds_by_index.get(pi) or [])
            got = set(new_p.get("unit_pool") or [])
            if got != expected:
                raise RuntimeError(
                    f"{name} pool {pi}: expected {sorted(expected)} got {sorted(got)}"
                )

        path.write_text(new_text, encoding="utf-8", newline="\n")
        changed_files += 1
        total_adds += len(applied)
        print(f"{name}: +{sorted(set(applied))} ({len(applied)} inserts)")

    print(f"\nUpdated {changed_files} files, {total_adds} unit inserts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
