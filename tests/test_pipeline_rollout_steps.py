"""Pipeline n_steps override for squad levels with large map observations."""

from train.async_train_universal import apply_pipeline_rollout_steps


def test_squad_level_takes_pipeline_n_steps():
    cfg = {
        "agent_architecture": "squad",
        "training": {"n_steps": 1024, "batch_size": 256},
    }
    apply_pipeline_rollout_steps(cfg, {"training": {"n_steps": 256}})
    assert cfg["training"]["n_steps"] == 256
    assert cfg["training"]["batch_size"] == 256


def test_universal_level_keeps_its_own_n_steps():
    cfg = {
        "agent_architecture": "universal",
        "training": {"n_steps": 2048},
    }
    apply_pipeline_rollout_steps(cfg, {"training": {"n_steps": 256}})
    assert cfg["training"]["n_steps"] == 2048


def test_missing_pipeline_n_steps_leaves_level_unchanged():
    cfg = {"agent_architecture": "squad", "training": {"n_steps": 1024}}
    apply_pipeline_rollout_steps(cfg, {})
    assert cfg["training"]["n_steps"] == 1024


def test_squad_level_takes_pipeline_target_kl_over_null():
    cfg = {
        "agent_architecture": "squad",
        "training": {"n_steps": 1024, "target_kl": None},
    }
    apply_pipeline_rollout_steps(
        cfg, {"training": {"n_steps": 256, "target_kl": 2.0}}
    )
    assert cfg["training"]["n_steps"] == 256
    assert cfg["training"]["target_kl"] == 2.0


def test_universal_level_keeps_its_own_target_kl():
    cfg = {"agent_architecture": "universal", "training": {"target_kl": None}}
    apply_pipeline_rollout_steps(cfg, {"training": {"n_steps": 256, "target_kl": 2.0}})
    assert cfg["training"]["target_kl"] is None
