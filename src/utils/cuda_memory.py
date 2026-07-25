"""Helpers to release GPU memory after PPO updates and model hot-swaps."""
from __future__ import annotations

import gc
from typing import Optional, Union

import torch
from loguru import logger


def release_ppo_minibatch_vram(
    device: Union[torch.device, str],
    *,
    empty_cache: bool = False,
    reset_peak: bool = False,
) -> None:
    """Drop cached CUDA blocks after a PPO update (PyTorch keeps freed pools otherwise)."""
    if not torch.cuda.is_available():
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    gc.collect()
    if empty_cache:
        torch.cuda.empty_cache()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(dev)


def clear_policy_optimizer(policy) -> None:
    """
    Drop Adam (or other) optimizer state.

    ``module.to('cpu')`` alone leaves optimizer.state tensors on GPU until the
    optimizer object is collected — a common source of opponent resync leaks.
    """
    if policy is None:
        return
    opt = getattr(policy, "optimizer", None)
    if opt is None:
        return
    try:
        policy.optimizer = None
    except Exception:
        pass
    try:
        opt.state.clear()
    except Exception:
        pass
    try:
        del opt
    except Exception:
        pass


def _clear_forward_cache(policy) -> None:
    fe = getattr(policy, "features_extractor", None)
    if fe is not None and hasattr(fe, "clear_forward_cache"):
        fe.clear_forward_cache()


def dispose_opponent_snapshot(model, *, empty_cache: bool = False) -> None:
    """Release an opponent snapshot (inference-only or full PPO).

    ``empty_cache`` defaults to False: vec-env workers share one GPU with the
    collector/learner; ``torch.cuda.empty_cache()`` in one process can TDR or
    poison sibling CUDA contexts (``unspecified launch failure``).
    """
    if model is None:
        return
    if getattr(model, "_inference_only", False):
        policy = getattr(model, "policy", None)
        if policy is not None:
            try:
                policy.set_training_mode(False)
            except Exception:
                pass
            _clear_forward_cache(policy)
            clear_policy_optimizer(policy)
            try:
                policy.to("cpu")
            except Exception:
                pass
            try:
                model.policy = None
            except Exception:
                pass
        gc.collect()
        if empty_cache and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        return
    dispose_ppo_model(model, empty_cache=empty_cache)


def dispose_ppo_model(model, *, empty_cache: bool = False) -> None:
    """Best-effort release of a MaskablePPO / opponent snapshot before replacing it."""
    if model is None:
        return
    if getattr(model, "_inference_only", False):
        dispose_opponent_snapshot(model, empty_cache=empty_cache)
        return
    try:
        if hasattr(model, "_last_lstm_states"):
            model._last_lstm_states = None
    except Exception:
        pass
    try:
        policy = getattr(model, "policy", None)
        if policy is not None:
            policy.set_training_mode(False)
            _clear_forward_cache(policy)
            clear_policy_optimizer(policy)
            policy.to("cpu")
    except Exception:
        pass
    try:
        if hasattr(model, "rollout_buffer"):
            buf = model.rollout_buffer
            if buf is not None:
                if hasattr(buf, "clear_vram_cache"):
                    buf.clear_vram_cache()
                if hasattr(buf, "close_memmaps"):
                    buf.close_memmaps()
            model.rollout_buffer = None
    except Exception:
        pass
    try:
        model.policy = None
    except Exception:
        pass
    del model
    gc.collect()
    if empty_cache and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def log_cuda_memory(device: Union[torch.device, str], label: str) -> None:
    if not torch.cuda.is_available():
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    alloc_mb = torch.cuda.memory_allocated(idx) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(idx) / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
    logger.info(
        f"{label}: cuda_alloc_mb={alloc_mb:.0f} reserved_mb={reserved_mb:.0f} "
        f"peak_mb={peak_mb:.0f}"
    )
