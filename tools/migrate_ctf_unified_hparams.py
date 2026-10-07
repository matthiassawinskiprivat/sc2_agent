"""Unify CTF squad training knobs and reward blocks.

Applies to every ``config_universal/*.yaml`` with ``agent_architecture: squad``:

* ``model.gamma`` → 0.99
* ``model.n_epochs`` → 4
* ``training.training_steps`` → half the current value
* ``env.max_steps`` → 100
* ``env.every_k_episodes`` → 200
* ``env.rewards`` → the canonical ``ctf_level_01`` economic block
  (drops per-level ``heal_cast_reward`` annealing and other divergences)

Idempotent. Run from the repo root.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

CANONICAL_GAMMA = 0.99
CANONICAL_N_EPOCHS = 4
CANONICAL_MAX_STEPS = 100
CANONICAL_EVERY_K = 200

# Same block as config_universal/ctf_level_01.yaml (economic shaping + outcomes).
CANONICAL_REWARDS = """\
  rewards:
    economic_vespene_weight: 1.5
    economic_value_norm: 100.0
    attacker_enemy_value_reward: 3.0
    attacker_own_value_penalty: 1.0
    defender_enemy_value_reward: 2.5
    defender_own_value_penalty: 3.0
    defender_pylon_penalty_scale: 1.5
    defender_retained_value_bonus: 4.0
    heal_value_reward: 4.0
    pylon_damage_reward: 2.0
    pylon_damage_penalty: 0.9
    melee_multiplier: 0.5
    distance_multiplier: 2.0
    defender_zone_radius: 10.0
    attacker_objective_radius: 12.0
    objective_credit_falloff: 15.0
    objective_credit_min_scale: 0.25
    defender_zone_leave_penalty: 0.08
    defender_zone_return_scale: 1.0
    cooldown_retreat_scale: 0.6
    aoe_land_reward: 0.75
    detection_reward: 1.0
    win_reward: 15.0
    loss_penalty: -12.5
    draw_penalty: -10.0
    step_penalty: -0.03
"""


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _replace_scalar(lines: list[str], key: str, value: str) -> tuple[list[str], bool]:
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:")
    out: list[str] = []
    changed = False
    for line in lines:
        match = pattern.match(line)
        if match is None:
            out.append(line)
            continue
        indent = match.group(1)
        new_line = f"{indent}{key}: {value}"
        if line.rstrip() != new_line:
            changed = True
        out.append(new_line)
    return out, changed


def _halve_training_steps(lines: list[str]) -> tuple[list[str], bool]:
    """Halve once. Already-halved values (post-migration set) are left alone."""
    # Pre-migration → post-migration. Re-running must not halve again.
    already_halved = {125_000, 250_000, 500_000}
    pattern = re.compile(r"^(\s*)training_steps\s*:\s*([0-9_]+)\s*$")
    out: list[str] = []
    changed = False
    for line in lines:
        match = pattern.match(line)
        if match is None:
            out.append(line)
            continue
        indent, raw = match.group(1), match.group(2)
        current = int(raw.replace("_", ""))
        if current in already_halved:
            out.append(line)
            continue
        half = current // 2
        rendered = f"{half:_}" if "_" in raw else str(half)
        new_line = f"{indent}training_steps: {rendered}"
        if line.rstrip() != new_line:
            changed = True
        out.append(new_line)
    return out, changed


def _replace_rewards_block(lines: list[str]) -> tuple[list[str], bool]:
    pattern = re.compile(r"^(\s*)rewards\s*:\s*$")
    canonical = CANONICAL_REWARDS.rstrip("\n").splitlines()
    out: list[str] = []
    i = 0
    changed = False
    while i < len(lines):
        match = pattern.match(lines[i])
        if match is None:
            out.append(lines[i])
            i += 1
            continue
        base = len(match.group(1))
        start = i
        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                break
            if _indent(lines[i]) <= base and not stripped.startswith("- "):
                break
            i += 1
        old_block = lines[start:i]
        # Drop documenting comments immediately above the rewards key.
        while out and out[-1].lstrip().startswith("#") and _indent(out[-1]) == base:
            out.pop()
        if old_block == canonical:
            out.extend(old_block)
        else:
            out.extend(canonical)
            changed = True
    return out, changed


def main() -> int:
    touched: list[str] = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        if path.name == "pipeline.yaml":
            continue
        original = path.read_text(encoding="utf-8")
        if "agent_architecture: squad" not in original:
            continue
        lines = original.splitlines()

        lines, c1 = _replace_scalar(lines, "gamma", str(CANONICAL_GAMMA))
        lines, c2 = _replace_scalar(lines, "n_epochs", str(CANONICAL_N_EPOCHS))
        lines, c3 = _halve_training_steps(lines)
        lines, c4 = _replace_scalar(lines, "max_steps", str(CANONICAL_MAX_STEPS))
        lines, c5 = _replace_scalar(lines, "every_k_episodes", str(CANONICAL_EVERY_K))
        lines, c6 = _replace_rewards_block(lines)

        if not (c1 or c2 or c3 or c4 or c5 or c6):
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        touched.append(path.name)

    print(f"Updated {len(touched)} CTF squad level file(s).")
    for name in touched:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
