"""Feature extractor and pointer network for CTF squad architecture."""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.components.entity_encoder import EntityEncoder, PointerAttention
from src.models.components.reasoning_core import ReasoningCore


class SquadNetwork(BaseFeaturesExtractor):
    """
    WHO: per-unit scalar keys via pointer attention (no GAP in key path).
    WHERE: global map + single hard-anchor tactical crop fused for spatial decoder.

    ``anchor_tactical`` is one (C, 32, 32) crop for the hard-selected unit
    (previous-step selection, or provisional first-alive). Soft blend over N
    unit crops is intentionally removed so obs/CNN cost stays O(1) in squad size.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 512,
        thinking_steps: int = 3,
        cnn_chunk_size: int = 16,
        tactical_cnn_chunk_size: Optional[int] = None,
        use_self_attention: bool = False,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim)
        # Legacy alias: older YAML used tactical_cnn_chunk_size for the unit-crop CNN.
        chunk = tactical_cnn_chunk_size if tactical_cnn_chunk_size is not None else cnn_chunk_size
        self._cnn_chunk_size = max(1, int(chunk))
        self._tactical_cnn_chunk_size = self._cnn_chunk_size
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("SquadNetwork requires Dict observation space")

        map_space = observation_space.spaces["map"]
        if "anchor_tactical" not in observation_space.spaces:
            raise KeyError(
                "SquadNetwork requires observation key 'anchor_tactical' "
                "(hard-anchor crop); 'unit_tactical' stacks are no longer supported"
            )
        tactical_space = observation_space.spaces["anchor_tactical"]
        self.command_dim = int(observation_space.spaces["command"].shape[0])
        self.race_dim = (
            int(observation_space.spaces["race"].shape[0])
            if "race" in observation_space.spaces
            else 0
        )
        self.upgrades_dim = (
            int(observation_space.spaces["upgrades"].shape[0])
            if "upgrades" in observation_space.spaces
            else 0
        )
        self.max_units = int(observation_space.spaces["unit_mask"].shape[0])
        self.unit_feat_dim = int(observation_space.spaces["unit_scalars"].shape[1])
        n_global = int(map_space.shape[0])
        n_tactical = int(tactical_space.shape[0])

        self.global_cnn = nn.Sequential(
            nn.Conv2d(n_global, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.tactical_cnn = nn.Sequential(
            nn.Conv2d(n_tactical, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Key path: unit_scalars only (no GAP concat). Hidden dim widened to 128
        # since the 31-feature input is richer than the old 4-scalar input.
        self.entity_encoder = EntityEncoder(
            input_dim=self.unit_feat_dim,
            embed_dim=64,
            hidden_dim=128,
            use_self_attention=use_self_attention,
        )
        self.pointer_attention = PointerAttention(
            query_dim=features_dim, embed_dim=64, use_sigmoid=False
        )
        # Query: global GAP + tactical GAP + command (+ optional race/upgrades trailing).
        # (Entity keys stay on the pointer path only — do not GAP-concat into keys.)
        self.query_linear = nn.Sequential(
            nn.Linear(
                64 + 64 + self.command_dim + self.race_dim + self.upgrades_dim,
                features_dim,
            ),
            nn.ReLU(),
        )
        self.reasoning = ReasoningCore(input_dim=features_dim, hidden_dim=features_dim)
        self.thinking_steps = thinking_steps
        self.fused_spatial_channels = 128

        self._pointer_logits: Optional[torch.Tensor] = None
        self._spatial_fused: Optional[torch.Tensor] = None

    def clear_forward_cache(self) -> None:
        """Drop stashed forward tensors so autograd intermediates can be freed."""
        self._pointer_logits = None
        self._spatial_fused = None

    def _cnn_forward(self, cnn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a CNN in sample chunks; checkpoint in train to cap activation VRAM."""
        n = int(x.shape[0])
        chunk = self._cnn_chunk_size
        if n <= chunk:
            return cnn(x)
        parts = []
        use_ckpt = bool(self.training and torch.is_grad_enabled())
        for start in range(0, n, chunk):
            sl = x[start : start + chunk]
            if use_ckpt:
                parts.append(torch_checkpoint(cnn, sl, use_reentrant=False))
            else:
                parts.append(cnn(sl))
        return torch.cat(parts, dim=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        map_obs = observations["map"]
        command_obs = observations["command"]
        anchor_tactical = observations["anchor_tactical"]
        unit_scalars = observations["unit_scalars"]
        unit_mask = observations["unit_mask"]

        # --- WHERE path: global + single hard-anchor tactical CNN ---
        global_16 = self._cnn_forward(self.global_cnn, map_obs)
        gap_g = global_16.mean(dim=(2, 3))
        tactical_16 = self._cnn_forward(self.tactical_cnn, anchor_tactical)
        gap_t = tactical_16.mean(dim=(2, 3))

        # --- WHO path: rich unit scalars only ---
        entity_embs = self.entity_encoder(unit_scalars, unit_mask)

        parts = [gap_g, gap_t, command_obs]
        if self.race_dim > 0:
            parts.append(observations["race"])
        if self.upgrades_dim > 0:
            parts.append(observations["upgrades"])
        combined = torch.cat(parts, dim=1)
        query = self.query_linear(combined)
        thought = self.reasoning(
            input_tensor=query, hidden_state=query, thinking_steps=self.thinking_steps
        )

        pointer_logits = self._pointer_logits_from_query(thought, entity_embs, unit_mask)
        self._pointer_logits = pointer_logits
        self._spatial_fused = torch.cat([global_16, tactical_16], dim=1)

        return thought

    def _pointer_logits_from_query(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = keys.size()
        query_expanded = query.unsqueeze(1).expand(b, n, -1)
        energy = torch.tanh(
            self.pointer_attention.W_q(query_expanded) + self.pointer_attention.W_k(keys)
        )
        logits = self.pointer_attention.V(energy).squeeze(-1)
        # Soft mask for padding — keep magnitude finite before action-logit stabilize.
        logits = logits.masked_fill(mask == 0, -50.0)
        return logits
