"""One-shot migration of CTF level YAMLs to the unit-pointer action contract.

Three mechanical edits, applied with line surgery so hand-written comments and
formatting survive:

1. ``heal_local`` -> ``heal_unit`` in ``action_heads`` / ``loud_command_heads``.
2. Drop ``heal_click_cells``: the unit-pointer heal names its patient by tag, so
   click-square width no longer exists.
3. Drop per-level ``active_unit_scalars`` blocks outside ``ctf_level_01``, which
   now declares the full catalog once for the whole curriculum.

Run from the repo root. Idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
CATALOG_LEVEL = "ctf_level_01.yaml"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _drop_block(lines: list[str], key: str) -> tuple[list[str], bool]:
    """Remove a ``key:`` mapping entry and its indented block (plus lead comments)."""
    out: list[str] = []
    i = 0
    changed = False
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:")
    while i < len(lines):
        match = pattern.match(lines[i])
        if match is None:
            out.append(lines[i])
            i += 1
            continue
        base = len(match.group(1))
        # Pull off comment lines immediately above that documented this key.
        while out and out[-1].lstrip().startswith("#") and _indent(out[-1]) == base:
            out.pop()
        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                break
            if _indent(lines[i]) <= base and not stripped.startswith("- "):
                break
            i += 1
        changed = True
    return out, changed


def _rename_heal_head(lines: list[str]) -> tuple[list[str], bool]:
    changed = False
    out = []
    for line in lines:
        if re.fullmatch(r"\s*-\s*heal_local\s*", line):
            out.append(line.replace("heal_local", "heal_unit"))
            changed = True
            continue
        out.append(line)
    return out, changed


def main() -> int:
    touched: list[str] = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        if path.name in ("pipeline.yaml",):
            continue
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()

        lines, renamed = _rename_heal_head(lines)
        lines, dropped_click = _drop_block(lines, "heal_click_cells")
        dropped_scalars = False
        if path.name != CATALOG_LEVEL:
            lines, dropped_scalars = _drop_block(lines, "active_unit_scalars")

        if not (renamed or dropped_click or dropped_scalars):
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        flags = [
            name
            for name, flag in (
                ("heal_unit", renamed),
                ("heal_click_cells", dropped_click),
                ("active_unit_scalars", dropped_scalars),
            )
            if flag
        ]
        touched.append(f"{path.name}: {', '.join(flags)}")

    print(f"Updated {len(touched)} level file(s):")
    for entry in touched:
        print(f"  {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
