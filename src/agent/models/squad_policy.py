"""Maskable recurrent policy for CTF squad pointer architecture."""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.type_aliases import Schedule

from src.agent.maskable_recurrent_ppo import MaskableRecurrentActorCriticPolicy

from src.agent.models.spatial_decoder import SpatialDecoder, tactical_center_spatial_bias
from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import MAX_SQUAD_UNITS
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def _stabilize_action_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)


class SquadRecurrentPolicy(MaskableRecurrentActorCriticPolicy):
    """Pointer + command + spatial heads for squad control."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        *args,
        **kwargs,
    ):
        self.spatial_resolution = 32
        kwargs.pop("spatial_resolution", None)
        kwargs.setdefault("features_extractor_class", SquadNetwork)
        fe_kw = dict(kwargs.get("features_extractor_kwargs") or {})
        fe_kw.setdefault("features_dim", 512)
        fe_kw.setdefault("thinking_steps", 3)
        kwargs["features_extractor_kwargs"] = fe_kw
        if "lstm_hidden_size" not in kwargs:
            kwargs["lstm_hidden_size"] = 512
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.command_head = nn.Linear(latent_dim_pi, FIXED_UNIVERSAL_COMMAND_DIM)
        self.spatial_decoder = SpatialDecoder(
            latent_dim_pi,
            fused_channels=128,
            spatial_resolution=self.spatial_resolution,
        )
        delattr(self, "action_net")
        if self.ortho_init:
            self.command_head.apply(partial(self.init_weights, gain=0.01))
            self.spatial_decoder.apply(partial(self.init_weights, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _expand_pointer_logits(self, pointer_logits: torch.Tensor) -> torch.Tensor:
        """Map scalar logits to independent binary categorical pairs."""
        zeros = torch.zeros_like(pointer_logits)
        pairs = torch.stack([zeros, pointer_logits], dim=-1)
        return pairs.reshape(pointer_logits.shape[0], -1)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        fe = self.features_extractor
        pointer_logits = getattr(fe, "_pointer_logits", None)
        if pointer_logits is None:
            pointer_logits = torch.zeros(
                latent_pi.shape[0], MAX_SQUAD_UNITS, device=latent_pi.device
            )
        fused = getattr(fe, "_spatial_fused", None)
        if fused is None:
            fused = torch.zeros(
                latent_pi.shape[0], 128, 16, 16,
                device=latent_pi.device, dtype=latent_pi.dtype,
            )
        spatial_logits = self.spatial_decoder(latent_pi, fused)
        bump = tactical_center_spatial_bias(self.spatial_resolution, latent_pi)
        spatial_logits = spatial_logits + bump.unsqueeze(0)
        cmd_logits = self.command_head(latent_pi)
        ptr_logits = self._expand_pointer_logits(pointer_logits)
        action_logits = torch.cat([ptr_logits, cmd_logits, spatial_logits], dim=-1)
        action_logits = _stabilize_action_logits(action_logits)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    @staticmethod
    def _sample_multidiscrete_safe(dist, *, deterministic: bool) -> torch.Tensor:
        """Sample actions; fall back to mode() when CUDA multinomial sees invalid probs."""
        if deterministic:
            return dist.get_actions(deterministic=True)
        parts = []
        for sub_dist in dist.distributions:
            probs = sub_dist.probs
            invalid = (~torch.isfinite(probs).all(dim=-1)) | (probs.sum(dim=-1) <= 1e-12)
            if not invalid.any():
                parts.append(sub_dist.sample())
                continue
            mode = sub_dist.mode()
            sampled = sub_dist.sample()
            parts.append(torch.where(invalid, mode, sampled))
        return torch.stack(parts, dim=1)

    def _disable_spatial_mask(self, action_masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if action_masks is None:
            return None
        masks = action_masks.clone()
        cmd_spatial_start = MAX_SQUAD_UNITS * 2 + FIXED_UNIVERSAL_COMMAND_DIM
        if masks.shape[-1] > cmd_spatial_start:
            masks[..., cmd_spatial_start:] = True
        return masks

    def forward(self, obs, lstm_states, episode_starts, action_masks=None, deterministic=False):
        action_masks = self._disable_spatial_mask(action_masks)
        features = self.extract_features(obs)
        pi_f, vf_f = (features, features) if self.share_features_extractor else features
        l_pi, s_pi = self._process_sequence(pi_f, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic:
            l_vf, s_vf = self._process_sequence(vf_f, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            l_vf, s_vf = l_pi.detach(), tuple(s.detach() for s in s_pi)
        else:
            l_vf, s_vf = self.critic(vf_f), s_pi

        l_pi, l_vf = self.mlp_extractor.forward_actor(l_pi), self.mlp_extractor.forward_critic(l_vf)
        values, dist = self.value_net(l_vf), self._get_action_dist_from_latent(l_pi)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        actions = self._sample_multidiscrete_safe(dist, deterministic=deterministic)
        return actions, values, dist.log_prob(actions), RNNStates(s_pi, s_vf)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts, action_masks=None):
        action_masks = self._disable_spatial_mask(action_masks)
        return super().evaluate_actions(obs, actions, lstm_states, episode_starts, action_masks)

    def get_distribution(self, obs, lstm_states, episode_starts, action_masks=None):
        action_masks = self._disable_spatial_mask(action_masks)
        return super().get_distribution(obs, lstm_states, episode_starts, action_masks)