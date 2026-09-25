"""Assign a gentle mineral/vespene budget ramp across the CTF pipeline order.

Previously, reordered phase-1 action lessons still carried late-curriculum
budgets (heal/ctf_18 at 1200–2300, drop at 2500+), while phase-2 layer lessons
dropped back to 400–700. That produced huge jumps and taught the wrong army
scale for early actions.

This script walks ``pipeline.yaml`` from the first CTF/scenario/AoE level and
sets attacker budgets on a smooth linear curve, with defenders at 75% (rounded
to 25). Idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config_universal"
PIPELINE = CONFIG_DIR / "pipeline.yaml"

# Attacker budgets at the first / last CTF curriculum step.
START_ATTACK_M, START_ATTACK_V = 200, 0
END_ATTACK_M, END_ATTACK_V = 1800, 1400
DEFENDER_RATIO = 0.75
ROUND_TO = 25


def _round_res(value: float) -> int:
    return int(ROUND_TO * round(float(value) / ROUND_TO))


def _ctf_pipeline_levels() -> list[str]:
    pipe = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    levels = list(pipe.get("levels") or [])
    start = next(
        (
            i
            for i, name in enumerate(levels)
            if str(name).startswith(("ctf_", "scenario_", "aoe_"))
        ),
        None,
    )
    if start is None:
        return []
    return [str(n) for n in levels[start:]]


def _attack_budget(index: int, n: int) -> tuple[int, int]:
    t = 0.0 if n <= 1 else index / (n - 1)
    m = _round_res(START_ATTACK_M + (END_ATTACK_M - START_ATTACK_M) * t)
    v = _round_res(START_ATTACK_V + (END_ATTACK_V - START_ATTACK_V) * t)
    return max(m, START_ATTACK_M), max(v, 0)


def _defend_budget(attack_m: int, attack_v: int) -> tuple[int, int]:
    return _round_res(attack_m * DEFENDER_RATIO), _round_res(attack_v * DEFENDER_RATIO)


def _pool_targets(pools: list[dict], attack_m: int, attack_v: int) -> list[tuple[int, int]]:
    def_m, def_v = _defend_budget(attack_m, attack_v)
    out: list[tuple[int, int]] = []
    for pool in pools:
        role = str(pool.get("high_level_command") or "attack").lower()
        if role.startswith("defend"):
            out.append((def_m, def_v))
        else:
            out.append((attack_m, attack_v))
    return out


def _apply_file(path: Path, targets: list[tuple[int, int]]) -> bool:
    text = path.read_text(encoding="utf-8")
    minerals = [m for m, _ in targets]
    vespene = [v for _, v in targets]
    m_iter = iter(minerals)
    v_iter = iter(vespene)

    def repl_m(match: re.Match[str]) -> str:
        try:
            return f"{match.group(1)}{next(m_iter)}"
        except StopIteration as exc:
            raise RuntimeError(f"{path.name}: more mineral_budget keys than pools") from exc

    def repl_v(match: re.Match[str]) -> str:
        try:
            return f"{match.group(1)}{next(v_iter)}"
        except StopIteration as exc:
            raise RuntimeError(f"{path.name}: more vespene_budget keys than pools") from exc

    new_text, n_m = re.subn(r"(mineral_budget:\s*)\d+", repl_m, text)
    new_text, n_v = re.subn(r"(vespene_budget:\s*)\d+", repl_v, new_text)
    if n_m != len(targets) or n_v != len(targets):
        raise RuntimeError(
            f"{path.name}: expected {len(targets)} budget pairs, "
            f"found mineral={n_m} vespene={n_v}"
        )
    # Ensure iterators were fully consumed.
    if list(m_iter) or list(v_iter):
        raise RuntimeError(f"{path.name}: unused budget targets left over")
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    levels = _ctf_pipeline_levels()
    if not levels:
        print("No CTF curriculum levels found in pipeline.yaml", file=sys.stderr)
        return 1

    n = len(levels)
    changed = 0
    print(f"{'idx':>3} {'level':55} {'atkM':>5} {'atkV':>5} {'defM':>5} {'defV':>5}")
    prev_m = None
    for i, name in enumerate(levels):
        path = CONFIG_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pools = list((cfg.get("env") or {}).get("unit_pools") or [])
        if not pools:
            print(f"skip {name}: no unit_pools", file=sys.stderr)
            continue
        atk_m, atk_v = _attack_budget(i, n)
        def_m, def_v = _defend_budget(atk_m, atk_v)
        jump = ""
        if prev_m is not None and atk_m - prev_m > ROUND_TO:
            jump = f"  +{atk_m - prev_m}"
        elif prev_m is not None and atk_m == prev_m:
            jump = "  (plateau)"
        print(f"{i:3d} {name:55} {atk_m:5d} {atk_v:5d} {def_m:5d} {def_v:5d}{jump}")
        targets = _pool_targets(pools, atk_m, atk_v)
        if _apply_file(path, targets):
            changed += 1
        prev_m = atk_m

    print(
        f"\nUpdated {changed}/{n} level files. "
        f"Ramp {START_ATTACK_M}/{START_ATTACK_V} -> {END_ATTACK_M}/{END_ATTACK_V}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
