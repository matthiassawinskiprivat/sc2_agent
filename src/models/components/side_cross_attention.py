"""Bidirectional cross-attention between allied and hostile entity embeddings."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _alive_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)
    return mask if mask.dtype == torch.bool else mask != 0


class SideCrossAttention(nn.Module):
    """Each side is the query; the other side supplies keys and values.

    Pre-norm ``MultiheadAttention`` plus a residual. The output projection is
    zero-initialised so a grafted checkpoint stays numerically identical until
    the new weights move.
    """

    def __init__(self, embed_dim: int = 64, nhead: int = 4):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by nhead {nhead}")
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=0.0, batch_first=True
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _one_direction(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        key_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q_alive = _alive_mask(query_mask, query)
        k_alive = _alive_mask(key_mask, key)
        can = q_alive.any(dim=-1) & k_alive.any(dim=-1)
        if not bool(can.any()):
            return query

        padding_mask = ~k_alive
        empty_keys = ~k_alive.any(dim=-1)
        if bool(empty_keys.any()):
            # Keep one dummy key legal so softmax stays defined on empty rows;
            # those rows are restored to the original query below.
            padding_mask = padding_mask.clone()
            padding_mask[empty_keys, 0] = False

        kn = self.norm_kv(key)
        attn_out, _ = self.attn(
            self.norm_q(query),
            kn,
            kn,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        out = query + self.out_proj(attn_out)
        out = out * q_alive.unsqueeze(-1).to(dtype=out.dtype)
        if bool((~can).any()):
            out = torch.where(can.view(-1, 1, 1), out, query)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        ally: torch.Tensor,
        enemy: torch.Tensor,
        ally_mask: Optional[torch.Tensor],
        enemy_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ally_out = self._one_direction(ally, enemy, ally_mask, enemy_mask)
        enemy_out = self._one_direction(enemy, ally, enemy_mask, ally_mask)
        return ally_out, enemy_out
