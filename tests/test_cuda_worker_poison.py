"""Tests for SubprocVecEnv CUDA poison detection / worker exit."""

import pytest


def test_is_cuda_context_poison_error_detects_unspecified_launch():
    from train.sb3_subproc_worker_safe import is_cuda_context_poison_error

    assert is_cuda_context_poison_error(
        RuntimeError("CUDA error: unspecified launch failure")
    )
    assert is_cuda_context_poison_error(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )
    assert not is_cuda_context_poison_error(RuntimeError("SC2 process died"))
    assert not is_cuda_context_poison_error(ValueError("bad action"))


def test_reset_retry_exits_on_cuda_poison(monkeypatch):
    from train import sb3_subproc_worker_safe as mod

    class _Env:
        observation_space = None

        def reset(self, **kwargs):
            raise RuntimeError("CUDA error: unspecified launch failure")

        def close(self):
            pass

    calls = {"exit": 0}

    def _fake_exit(code):
        calls["exit"] = code
        raise SystemExit(code)

    monkeypatch.setattr(mod.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit) as ei:
        mod._reset_env_with_retry(_Env())
    assert ei.value.code == 1
    assert calls["exit"] == 1
