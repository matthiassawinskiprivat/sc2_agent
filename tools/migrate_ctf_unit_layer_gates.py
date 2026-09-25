"""Align CTF unit pools with map-layer gates — keep pipeline level order.

- Flying / air-DPS-base channels are introduced at ``ctf_level_24`` (transport),
  because Medivac / Warp Prism / Overlord Transport need them.
- Massive overlay channels stay at ``ctf_level_07``.
- Levels before a gate lose units that require that gate's layers.
- ``ctf_heal_thor_scv_vs_thor_scv`` keeps Thor (repair intro; overlays not the lesson).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "config_universal"
PIPELINE = ROOT / "pipeline.yaml"

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
KEEP_MASSIVE = {"ctf_heal_thor_scv_vs_thor_scv"}

FLYING_LAYERS = [
    "allied_unit_flying",
    "enemy_unit_flying",
    "allied_a_dps_base",
    "enemy_a_dps_base",
]
_UNIT_LINE = re.compile(r"^(\s+)-\s+([A-Z][A-Z0-9_]*)\s*$")


def _strip_unit_lines(text: str, remove: set[str]) -> tuple[str, list[str]]:
    if not remove:
        return text, []
    removed: list[str] = []
    out: list[str] = []
    in_pool = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" ")
        if stripped.startswith("unit_pool:"):
            in_pool = True
            out.append(line)
            continue
        if in_pool:
            m = _UNIT_LINE.match(line.rstrip("\n\r"))
            if m:
                unit = m.group(2)
                if unit in remove:
                    removed.append(unit)
                    continue
                out.append(line)
                continue
            in_pool = False
        out.append(line)
    return "".join(out), removed


def _ensure_layers(text: str, key: str, layers: list[str]) -> tuple[str, bool]:
    """Append missing layer names under ``active_layers`` / ``active_tactical_layers``."""
    cfg = yaml.safe_load(text) or {}
    env = cfg.get("env") or {}
    existing = list(env.get(key) or [])
    missing = [L for L in layers if L not in existing]
    if not missing:
        return text, False

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_block = False
    block_indent = "  "
    item_indent = "  - "
    inserted = False
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith(f"{key}:"):
            in_block = True
            block_indent = indent
            item_indent = f"{indent}  - " if indent else "  - "
            # peek existing item indent
            for j in range(i + 1, min(i + 5, len(lines))):
                m = re.match(r"^(\s+)-\s+", lines[j])
                if m:
                    item_indent = f"{m.group(1)}- "
                    break
            out.append(line)
            continue
        if in_block:
            if stripped.startswith("- "):
                out.append(line)
                continue
            # left the list — insert missing before this line
            if not inserted:
                for L in missing:
                    out.append(f"{item_indent}{L}\n")
                inserted = True
            in_block = False
        out.append(line)
    if in_block and not inserted:
        for L in missing:
            out.append(f"{item_indent}{L}\n")
    return "".join(out), True


def main() -> int:
    levels = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))["levels"]
    fly_gate = "ctf_level_24"
    mass_gate = "ctf_level_07"
    fly_i = levels.index(fly_gate)
    mass_i = levels.index(mass_gate)

    # 1) Attach flying map channels to the transport intro.
    path24 = ROOT / f"{fly_gate}.yaml"
    text24 = path24.read_text(encoding="utf-8")
    text24, ch_g = _ensure_layers(text24, "active_layers", FLYING_LAYERS)
    text24, ch_t = _ensure_layers(text24, "active_tactical_layers", FLYING_LAYERS)
    if ch_g or ch_t:
        path24.write_text(text24, encoding="utf-8", newline="\n")
        print(f"{fly_gate}: added flying / air-DPS-base layers")
    else:
        print(f"{fly_gate}: flying layers already present")

    # 2) Strip gated units from earlier levels.
    for name in levels:
        if not name.startswith(("ctf_", "scenario_", "aoe_")):
            continue
        path = ROOT / f"{name}.yaml"
        if not path.exists():
            continue
        idx = levels.index(name)
        remove: set[str] = set()
        if idx < fly_i:
            remove |= FLYING
        if idx < mass_i and name not in KEEP_MASSIVE:
            remove |= MASSIVE
        if not remove:
            continue
        text = path.read_text(encoding="utf-8")
        new_text, removed = _strip_unit_lines(text, remove)
        if removed:
            path.write_text(new_text, encoding="utf-8", newline="\n")
            print(f"{name}: removed {sorted(set(removed))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
