"""All CTF squad levels share one training/reward contract."""
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

_CANONICAL_REWARDS = {
    "economic_vespene_weight": 1.5,
    "economic_value_norm": 100.0,
    "attacker_enemy_value_reward": 3.0,
    "attacker_own_value_penalty": 1.0,
    "defender_enemy_value_reward": 2.5,
    "defender_own_value_penalty": 3.0,
    "defender_pylon_penalty_scale": 1.5,
    "defender_retained_value_bonus": 4.0,
    "heal_value_reward": 4.0,
    "pylon_damage_reward": 2.0,
    "pylon_damage_penalty": 0.9,
    "melee_multiplier": 0.5,
    "distance_multiplier": 2.0,
    "defender_zone_radius": 10.0,
    "attacker_objective_radius": 12.0,
    "objective_credit_falloff": 15.0,
    "objective_credit_min_scale": 0.25,
    "defender_zone_leave_penalty": 0.08,
    "defender_zone_return_scale": 1.0,
    "cooldown_retreat_scale": 0.6,
    "aoe_land_reward": 0.75,
    "detection_reward": 1.0,
    "win_reward": 15.0,
    "loss_penalty": -12.5,
    "draw_penalty": -10.0,
    "step_penalty": -0.03,
}


def _squad_levels():
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        if path.name == "pipeline.yaml":
            continue
        text = path.read_text(encoding="utf-8")
        if "agent_architecture: squad" not in text:
            continue
        yield path, yaml.safe_load(text)


def test_ctf_squad_levels_share_unified_hparams_and_rewards():
    for path, data in _squad_levels():
        env = data["env"]
        model = data["model"]
        assert model.get("gamma") == 0.99, path.name
        assert model.get("n_epochs") == 4, path.name
        assert env.get("max_steps") == 100, path.name
        assert env.get("every_k_episodes") == 200, path.name
        assert env.get("rewards") == _CANONICAL_REWARDS, path.name
        assert "heal_cast_reward" not in (env.get("rewards") or {}), path.name


def test_ctf_squad_training_steps_are_the_halved_budget():
    """Post-unify budgets are half the pre-unify values (idempotent set)."""
    allowed = {125_000, 250_000, 500_000}
    for path, data in _squad_levels():
        steps = data["training"]["training_steps"]
        assert steps in allowed, f"{path.name} training_steps={steps}"


def test_pipeline_levels_after_ctf_level_23_use_250k_steps():
    pipeline = yaml.safe_load((_CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    names = list(pipeline["levels"])
    after_idx = names.index("ctf_level_23") + 1
    for name in names[after_idx:]:
        path = _CONFIG_DIR / f"{name}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        steps = data["training"]["training_steps"]
        assert steps in {125_000, 250_000}, f"{name} training_steps={steps}"
