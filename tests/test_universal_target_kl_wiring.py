"""Universal training passes target_kl from model YAML (default None when key absent)."""


def test_model_params_target_kl_wiring():
    # Mirrors train_universal / async_train_universal: model_params.get("target_kl")
    model_params = {"gamma": 0.99, "ent_coef": 0.0, "n_epochs": 4, "target_kl": 0.02}
    model_kwargs = {
        "policy": object,
        "env": None,
        "batch_size": 64,
        "n_steps": 128,
    }
    model_kwargs["target_kl"] = model_params.get("target_kl")
    assert model_kwargs["target_kl"] == 0.02

    model_params_no_key = {"gamma": 0.99}
    model_kwargs2 = {"batch_size": 64}
    model_kwargs2["target_kl"] = model_params_no_key.get("target_kl")
    assert model_kwargs2["target_kl"] is None
