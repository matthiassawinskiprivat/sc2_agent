"""Rewrite CTF/scenario/AoE env_mode to accumulate by pipeline intro order.

Modes stick once introduced:
  heal        @ ctf_heal_queen_vs_queen
  cloak       @ ctf_level_21
  spellcaster @ aoe_high_templar_zealot_vs_marine_marauder
  drop        @ ctf_level_24
  upgrades    @ ctf_level_27
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "config_universal"
PIPELINE = ROOT / "pipeline.yaml"

INTRO = {
    "heal": "ctf_heal_queen_vs_queen",
    "cloak": "ctf_level_21",
    "spellcaster": "aoe_high_templar_zealot_vs_marine_marauder",
    "drop": "ctf_level_24",
    "upgrades": "ctf_level_27",
}
ORDER = ("heal", "cloak", "spellcaster", "drop", "upgrades")


def _expected_mode(levels: list[str], name: str) -> str | None:
    idx = levels.index(name)
    modes = [m for m in ORDER if idx >= levels.index(INTRO[m])]
    return ",".join(modes) if modes else None


def _set_env_mode(text: str, mode: str | None) -> str:
    """Insert/replace/remove env.env_mode while preserving surrounding YAML."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_env = False
    env_indent = ""
    mode_indent = "  "
    saw_mode = False
    inserted = False

    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("env:") and not indent:
            in_env = True
            env_indent = ""
            out.append(line)
            continue

        if in_env:
            # Left the env block (next top-level key).
            if stripped and not line.startswith(" ") and not line.startswith("\t"):
                if mode is not None and not saw_mode and not inserted:
                    out.append(f"{mode_indent}env_mode: {mode}\n")
                    inserted = True
                in_env = False
                out.append(line)
                continue

            if stripped.startswith("env_mode:"):
                saw_mode = True
                if mode is None:
                    continue  # drop the line
                # Keep sibling indent from the existing line.
                mode_indent = indent or "  "
                out.append(f"{mode_indent}env_mode: {mode}\n")
                continue

            # Prefer placing env_mode after include_race / alternate_trainee_race
            # / every_k / headless cluster — insert before unit_pools if missing.
            if (
                mode is not None
                and not saw_mode
                and not inserted
                and stripped.startswith("unit_pools:")
            ):
                out.append(f"{mode_indent}env_mode: {mode}\n")
                inserted = True

        out.append(line)

    if in_env and mode is not None and not saw_mode and not inserted:
        out.append(f"{mode_indent}env_mode: {mode}\n")

    return "".join(out)


def main() -> int:
    pipe = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    levels = pipe["levels"]
    ctf_start = next(
        i
        for i, l in enumerate(levels)
        if l.startswith("ctf_") or l.startswith("scenario_") or l.startswith("aoe_")
    )
    changed = 0
    for name in levels[ctf_start:]:
        path = ROOT / f"{name}.yaml"
        text = path.read_text(encoding="utf-8")
        expected = _expected_mode(levels, name)
        new_text = _set_env_mode(text, expected)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8", newline="\n")
            changed += 1
            print(f"{name}: {expected or '(none)'}")
        else:
            # Still print for visibility when already correct.
            cfg = yaml.safe_load(text) or {}
            got = (cfg.get("env") or {}).get("env_mode")
            print(f"{name}: unchanged ({got or '(none)'})")
    print(f"\nUpdated {changed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
