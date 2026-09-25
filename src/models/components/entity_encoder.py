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
            # Accept bool or float masks; treat anything != 0 as alive.
            alive = mask if mask.dtype == torch.bool else mask != 0
            mask_expanded = alive.unsqueeze(-1).to(dtype=embeddings.dtype)
            embeddings = embeddings * mask_expanded

        if self.use_self_attention:
            # TransformerEncoderLayer src_key_padding_mask convention:
            # True = IGNORE that position (opposite of our mask convention).
            #
            # Fully-masked rows (no allies / no visible enemies) make every key
            # padded. In ``eval`` mode that path yields NaN attention outputs
            # with trained weights; ``NaN * 0`` stays NaN and poisons the squad
            # query / GRU hidden states written into the rollout buffer.
            if mask is None:
                embeddings = self.self_attn(embeddings)
            else:
                fully_masked = ~alive.any(dim=-1)
                if bool(fully_masked.all()):
                    # Nothing to attend over — keep the zeroed MLP embeddings.
                    pass
                else:
                    padding_mask = ~alive
                    if bool(fully_masked.any()):
                        # Keep one dummy key legal so softmax stays defined, then
                        # zero those rows after attention.
                        padding_mask = padding_mask.clone()
                        padding_mask[fully_masked, 0] = False
                    embeddings = self.self_attn(
                        embeddings, src_key_padding_mask=padding_mask
                    )
                    embeddings = embeddings * mask_expanded
                    if bool(fully_masked.any()):
                        embeddings = embeddings.clone()
                        embeddings[fully_masked] = 0

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
