"""Print the additive env lists each pipeline level contributes (debug helper)."""

import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
KEYS = (
    "action_heads",
    "categorical_heads",
    "active_layers",
    "active_tactical_layers",
    "active_unit_scalars",
)


def main() -> int:
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    for level in pipeline.get("levels", []):
        path = CONFIG_DIR / f"{level}.yaml"
        if not path.exists():
            print(f"{level}: MISSING")
            continue
        conf = yaml.safe_load(path.read_text()) or {}
        env = conf.get("env") or {}
        arch = conf.get("agent_architecture", pipeline.get("agent_architecture"))
        bits = [f"arch={arch}"]
        for key in KEYS:
            val = env.get(key)
            if val:
                bits.append(f"{key}={val}")
        for flag in ("env_mode", "include_race", "include_upgrades", "damaged_units",
                     "heal_click_cells", "loud_command_heads", "alternate_trainee_race"):
            if flag in env:
                bits.append(f"{flag}={env[flag]}")
        print(f"{level}:\n    " + "\n    ".join(bits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
