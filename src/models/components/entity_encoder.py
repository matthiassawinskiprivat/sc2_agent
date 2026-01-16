import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityEncoder(nn.Module):
    """
    Encodes a variable number of entities (units) into embeddings.
    Used by Build Agent (for Workers) and Manager Agent (for Squads).
    """
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Shared MLP applied to every entity independently
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, Max_Entities, Input_Dim]
            mask: [Batch, Max_Entities] (1=Valid, 0=Padding)
            
        Returns:
            embeddings: [Batch, Max_Entities, Embed_Dim]
        """
        # Apply MLP to each entity
        embeddings = self.encoder(x)
        
        # Zero out padding to avoid noise
        if mask is not None:
            # mask is [B, N], expand to [B, N, Embed_Dim]
            mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
            embeddings = embeddings * mask_expanded
            
        return embeddings

class PointerAttention(nn.Module):
    """
    Selects one entity from the list based on a Query (Context).
    """
    def __init__(self, query_dim: int, embed_dim: int, hidden_dim: int = 64, use_sigmoid: bool = False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        
        # Project Query and Key (Entity Embedding) into same space
        self.W_q = nn.Linear(query_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1) # Scoring vector
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [Batch, Query_Dim] (The Command/Context)
            keys: [Batch, Max_Entities, Embed_Dim] (The Encoded Entities)
            mask: [Batch, Max_Entities] (1=Valid, 0=Padding)
            
        Returns:
            scores: [Batch, Max_Entities] (Normalized probabilities/scores)
        """
        # 1. Expand Query to match number of entities
        # [Batch, 1, Query_Dim] -> [Batch, N, Query_Dim]
        B, N, _ = keys.size()
        query_expanded = query.unsqueeze(1).expand(B, N, -1)
        
        # 2. Calculate Energy
        # Bahdanau Attention style: score = V * tanh(Wq*Q + Wk*K)
        energy = torch.tanh(self.W_q(query_expanded) + self.W_k(keys))
        logits = self.V(energy).squeeze(-1) # [Batch, N]
        
        # 3. Masking (Very important: don't select padding or dead units)
        if mask is not None:
            # For Softmax, we use -inf. For Sigmoid, a large negative number works too.
            logits = logits.masked_fill(mask == 0, -1e9)
            
        # 4. Normalization
        if self.use_sigmoid:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)
