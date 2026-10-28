"""One-shot migration of CTF reward blocks from hit-point to economic shaping.

Replaces the flat per-hit-point / flat-per-kill keys with role-specific,
resource-weighted coefficients:

    damage_reward   -> attacker_enemy_value_reward / defender_enemy_value_reward
    damage_penalty  -> attacker_own_value_penalty / defender_own_value_penalty
    kill_reward     -> (gone: a death now pays the value still standing)
    kill_penalty    -> (gone: same reason, on the losing side)
    heal_reward     -> heal_value_reward
    defender_step_reward -> defender_retained_value_bonus (terminal, not per step)

The old keys were "per hit point": ``damage_reward: 0.4`` paid 0.4 per point of
enemy health. The new keys are "per unit of normalised economic value", where one
value unit is ``economic_value_norm`` (default 100) worth of
``minerals + 1.5 × vespene``. Killing a full-health Marine (50/0) therefore pays
``0.5 × attacker_enemy_value_reward`` instead of ``45 × damage_reward``, so the
coefficients are scaled up to keep episode returns in a comparable range.

``melee_multiplier`` is kept: it still scales the economic coefficients for squads
that have to walk into range. ``pylon_damage_reward`` / ``pylon_damage_penalty`` /
``win_reward`` / ``loss_penalty`` / ``draw_penalty`` are untouched, so outcomes stay
dominant over shaping.

Run from the repo root. Idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

# Old per-hit-point keys, removed outright.
DROP_KEYS = (
    "damage_reward",
    "damage_penalty",
    "kill_reward",
    "kill_penalty",
    "heal_reward",
    "defender_step_reward",
)

# New economic block, inserted at the top of every CTF ``rewards:`` mapping.
# Attacker: strong enemy-value reward, deliberately small own-loss penalty, so a
# favourable or simply necessary trade into the pylon stays learnable.
# Defender: heavy own-value preservation, zone-gated enemy credit, heavier pylon
# penalty, and a terminal retained-value bonus instead of a per-step payout.
ECONOMIC_BLOCK = """\
    economic_vespene_weight: 1.5
    economic_value_norm: 100.0
    attacker_enemy_value_reward: 3.0
    attacker_own_value_penalty: 2.0
    defender_enemy_value_reward: 2.5
    defender_own_value_penalty: 3.0
    defender_pylon_penalty_scale: 1.5
    defender_retained_value_bonus: 4.0
    heal_value_reward: 4.0
"""

NEW_KEYS = tuple(
    line.split(":", 1)[0].strip() for line in ECONOMIC_BLOCK.strip().splitlines()
)


def _is_ctf_level(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "agent_architecture: squad" in text


def _drop_keys(lines: list[str]) -> tuple[list[str], bool]:
    pattern = re.compile(rf"^\s*(?:{'|'.join(DROP_KEYS)})\s*:")
    kept = [line for line in lines if not pattern.match(line)]
    return kept, len(kept) != len(lines)


def _insert_economic_block(lines: list[str]) -> tuple[list[str], bool]:
    if any(line.strip().startswith(NEW_KEYS[0] + ":") for line in lines):
        return lines, False
    for idx, line in enumerate(lines):
        if re.fullmatch(r"\s*rewards\s*:\s*", line):
            block = ECONOMIC_BLOCK.rstrip("\n").splitlines()
            return lines[: idx + 1] + block + lines[idx + 1 :], True
    return lines, False


def main() -> int:
    touched: list[str] = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        if path.name == "pipeline.yaml" or not _is_ctf_level(path):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        lines, dropped = _drop_keys(lines)
        lines, inserted = _insert_economic_block(lines)
        if not (dropped or inserted):
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        touched.append(path.name)

    print(f"Migrated {len(touched)} CTF level file(s) to economic rewards.")
    for name in touched:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
