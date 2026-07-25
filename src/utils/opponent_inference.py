"""Lightweight opponent policy loading for CTF self-play (inference only, no full PPO)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from stable_baselines3.common.utils import get_schedule_fn


@dataclass
class OpponentInferenceModel:
    """
    Policy-only snapshot for opponent ``predict`` in vec-env workers.

    Avoids constructing a full SB3 PPO object (optimizer, rollout buffer, env).
    """

    policy: torch.nn.Module
    _inference_only: bool = True

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks=None,
    ):
        return self.policy.predict(
            observation,
            state,
            episode_start,
            deterministic,
            action_masks=action_masks,
        )


def wire_maskable_action_dist(policy, action_space) -> None:
    """Match MaskableRecurrentPPO._setup_model action_dist wiring for inference."""
    import gymnasium as gym
    from sb3_contrib.common.maskable.distributions import (
        MaskableBernoulliDistribution,
        MaskableCategoricalDistribution,
        MaskableMultiCategoricalDistribution,
    )

    if isinstance(action_space, gym.spaces.Discrete):
        policy.action_dist = MaskableCategoricalDistribution(action_space.n)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        policy.action_dist = MaskableMultiCategoricalDistribution(action_space.nvec)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        policy.action_dist = MaskableBernoulliDistribution(action_space.n)


def build_opponent_policy(
    policy_class: Any,
    observation_space: Any,
    action_space: Any,
    policy_kwargs: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Construct a policy network for inference (no Adam optimizer state on GPU)."""
    from src.utils.cuda_memory import clear_policy_optimizer

    pk = dict(policy_kwargs)
    lr_schedule = get_schedule_fn(0.0)
    policy = policy_class(observation_space, action_space, lr_schedule, **pk)
    wire_maskable_action_dist(policy, action_space)
    clear_policy_optimizer(policy)
    policy.set_training_mode(False)
    return policy.to(device)


def load_opponent_inference_checkpoint(
    model_path: str,
    *,
    agent_architecture: str,
    observation_space: Any,
    action_space: Any,
    policy_class: Any,
    policy_kwargs: dict,
    rnn_type: str,
    device: torch.device,
    grid_size: int = 32,
) -> OpponentInferenceModel:
    """
    Load checkpoint weights into a policy-only opponent snapshot.

    Tries a strict policy state-dict load first; falls back to curriculum graft
    (squad or universal) when shapes differ.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    from src.utils.model_utils import (
        graft_model_weights,
        graft_squad_policy_from_universal,
        load_policy_weights_from_zip,
    )

    dev = torch.device(device)
    policy = build_opponent_policy(
        policy_class, observation_space, action_space, policy_kwargs, dev
    )

    ok, missing, unexpected = load_policy_weights_from_zip(
        policy, model_path, dev, strict=True
    )
    if ok and not missing:
        logger.debug(f"Opponent inference: strict load from {model_path!r}")
        return OpponentInferenceModel(policy=policy)

    if unexpected:
        logger.debug(
            f"Opponent inference: strict load had unexpected={len(unexpected)} "
            f"from {model_path!r}; trying graft"
        )

    _, params, _ = load_from_zip_file(model_path, device="cpu", load_data=False)
    saved_sd = params.get("policy")
    if saved_sd is None:
        raise ValueError(f"Checkpoint {model_path!r} has no policy weights")

    if agent_architecture == "squad":
        new_in = None
        if hasattr(observation_space, "spaces") and "map" in observation_space.spaces:
            new_in = int(observation_space.spaces["map"].shape[0])
        grafted = graft_squad_policy_from_universal(
            saved_sd,
            policy.state_dict(),
            new_in_channels=new_in,
        )
    else:
        if hasattr(observation_space, "spaces") and "map" in observation_space.spaces:
            new_in_channels = int(observation_space.spaces["map"].shape[0])
        else:
            new_in_channels = int(observation_space.shape[0])
        new_action_dim = getattr(action_space, "n", None)
        if new_action_dim is None and hasattr(action_space, "nvec"):
            new_action_dim = int(np.sum(action_space.nvec))
        grafted = graft_model_weights(
            saved_sd,
            new_in_channels=new_in_channels,
            new_action_dim=new_action_dim,
            new_grid_size=grid_size,
            target_state_dict=policy.state_dict(),
        )

    load_missing, load_unexpected = policy.load_state_dict(grafted, strict=False)
    logger.info(
        f"Opponent inference graft from {model_path!r}: "
        f"missing={len(load_missing)} unexpected={len(load_unexpected)}"
    )
    return OpponentInferenceModel(policy=policy)
