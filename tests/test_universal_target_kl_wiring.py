"""Universal training passes target_kl from training or model YAML (default None)."""


def _resolve_target_kl(training_params, model_params):
    # Mirrors train_universal / async_train_universal
    return training_params.get("target_kl", model_params.get("target_kl"))


def test_target_kl_from_training_section():
    training_params = {"learning_rate": 3e-4, "target_kl": 0.02}
    model_params = {"gamma": 0.99, "ent_coef": 0.0, "n_epochs": 4}
    assert _resolve_target_kl(training_params, model_params) == 0.02


def test_target_kl_from_model_section_fallback():
    training_params = {"learning_rate": 3e-4}
    model_params = {"gamma": 0.99, "target_kl": 0.015}
    assert _resolve_target_kl(training_params, model_params) == 0.015


def test_target_kl_training_null_wins_over_model():
    training_params = {"target_kl": None}
    model_params = {"target_kl": 0.02}
    assert _resolve_target_kl(training_params, model_params) is None


def test_target_kl_default_none():
    assert _resolve_target_kl({}, {}) is None
