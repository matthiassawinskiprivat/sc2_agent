import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityEncoder(nn.Module):
    """Encodes a variable number of entities (units) into per-unit embeddings.

    Each entity is first projected independently through a shared MLP.  When
    ``use_self_attention=True``, a single Transformer encoder layer then lets
    every entity attend to its squad-mates, providing relational context that
    the per-unit MLP cannot capture on its own.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int = 64,
        use_self_attention: bool = False,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.use_self_attention = use_self_attention
        if use_self_attention:
            # Pre-norm Transformer layer for training stability.
            # batch_first=True keeps tensors in [B, N, D] convention throughout.
            self.self_attn = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    [Batch, Max_Entities, Input_Dim]
            mask: [Batch, Max_Entities] — 1 = valid unit, 0 = padding

        Returns:
            embeddings: [Batch, Max_Entities, Embed_Dim]
        """
        embeddings = self.encoder(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
            embeddings = embeddings * mask_expanded

        if self.use_self_attention:
            # TransformerEncoderLayer src_key_padding_mask convention:
            # True = IGNORE that position (opposite of our mask convention).
            padding_mask = (mask == 0) if mask is not None else None
            embeddings = self.self_attn(embeddings, src_key_padding_mask=padding_mask)
            # Re-zero padding slots that attention may have polluted.
            if mask is not None:
                embeddings = embeddings * mask_expanded

        return embeddings


class PointerAttention(nn.Module):
    """Selects one entity from a list based on a query (context) vector."""

    def __init__(self, query_dim: int, embed_dim: int, hidden_dim: int = 64, use_sigmoid: bool = False):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.W_q = nn.Linear(query_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [Batch, Query_Dim]
            keys:  [Batch, Max_Entities, Embed_Dim]
            mask:  [Batch, Max_Entities] — 1 = valid, 0 = padding

        Returns:
            scores: [Batch, Max_Entities]
        """
        B, N, _ = keys.size()
        query_expanded = query.unsqueeze(1).expand(B, N, -1)

        energy = torch.tanh(self.W_q(query_expanded) + self.W_k(keys))
        logits = self.V(energy).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        if self.use_sigmoid:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)
