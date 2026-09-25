"""On-policy contract: rollout_policy.zip must match collection-time weights."""
from pathlib import Path


def test_collector_does_not_reload_weights_during_or_after_fill_before_signal():
    """
    Regression: mid/post-fill try_reload() before rollout_policy save made the
    learner load weights newer than old_log_prob (approx_kl spikes, abs_err ≫ 0).
    """
    src = Path("train/async_train_universal.py").read_text(encoding="utf-8")
    marker = "while filled < n_rollout_steps:"
    idx = src.find(marker)
    assert idx != -1
    # From fill loop through signal_ready — no try_reload.
    signal_idx = src.find("current_buffer.signal_ready", idx)
    assert signal_idx != -1
    fill_region = src[idx:signal_idx]
    assert "weight_sync.try_reload" not in fill_region
    assert "fill_frozen = True" in src[:idx]
    assert "save_rollout_policy_atomic" in fill_region or "save_rollout_policy_atomic" in src[idx:signal_idx]


def test_collector_saves_rollout_policy_after_fill_before_signal():
    """Zip must be published after the fill so it cannot lag behind collect weights."""
    src = Path("train/async_train_universal.py").read_text(encoding="utf-8")
    fill_loop = src.find("while filled < n_rollout_steps:")
    assert fill_loop != -1
    save_call = src.find("save_rollout_policy_atomic", fill_loop)
    signal = src.find("current_buffer.signal_ready", fill_loop)
    assert save_call != -1 and signal != -1
    assert fill_loop < save_call < signal
    assert "rollout_policy_checksum" in src[save_call:signal + 200]


def test_maybe_poll_noops_when_fill_frozen(monkeypatch):
    import train.async_train_universal as mod
    from unittest.mock import MagicMock

    sync = MagicMock()
    sync.fill_frozen = True
    sync.try_reload = MagicMock(return_value=True)
    state = {"last_poll": 0.0}
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 100.0)
    mod.maybe_poll_collector_weights(sync, None, None, state)
    sync.try_reload.assert_not_called()
