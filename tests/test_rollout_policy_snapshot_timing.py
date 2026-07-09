"""On-policy contract: rollout_policy.zip must match collection-time weights."""
from pathlib import Path


def test_collector_does_not_reload_weights_after_rollout_before_gae():
    """
    Regression: post-fill try_reload() before rollout_policy save made the learner load
    weights newer than old_log_prob (approx_kl ~6+, rollout_log_prob_abs_err ~7+).
    """
    src = Path("train/async_train_universal.py").read_text(encoding="utf-8")
    marker = "filled += rows_written"
    idx = src.find(marker)
    assert idx != -1
    after_fill = src[idx : idx + 600]
    assert "weight_sync.try_reload" not in after_fill
    assert "rollout_policy_path" not in after_fill


def test_collector_saves_rollout_policy_before_fill_loop():
    src = Path("train/async_train_universal.py").read_text(encoding="utf-8")
    pre_fill_reload = src.find("weight_sync.try_reload(model, vec_env)")
    assert pre_fill_reload != -1
    snapshot = src.find("rollout_policy_path", pre_fill_reload)
    fill_loop = src.find("while filled < n_rollout_steps:", pre_fill_reload)
    assert snapshot != -1 and fill_loop != -1
    assert snapshot < fill_loop
