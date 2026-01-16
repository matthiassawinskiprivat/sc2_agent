import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from src.models.components.entity_encoder import EntityEncoder, PointerAttention
from src.models.components.reasoning_core import ReasoningCore

class BuildNetwork(nn.Module):
    """
    The Brain of the Build Agent.
    Selects a worker and a precise build location.
    """
    def __init__(self, 
                 map_channels: int = 5,  # Local map features (Pathing, Height, Units...)
                 worker_feat_dim: int = 5, # x, y, health, idle, load
                 cmd_embed_dim: int = 32,
                 hidden_dim: int = 128,
                 map_resolution: int = 16):
        super().__init__()
        
        self.map_res = map_resolution
        self.hidden_dim = hidden_dim
        
        # 1. Encoders
        # Command Encoder (What to build?)
        self.cmd_embedding = nn.Embedding(200, cmd_embed_dim) # ~200 UnitTypeIds
        self.cmd_fc = nn.Linear(cmd_embed_dim + 2, 64) # +2 for Target (x,y)
        
        # Map Encoder (Local Grid)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * map_resolution * map_resolution, 128),
            nn.ReLU()
        )
        
        # Entity Encoder (Workers)
        self.worker_encoder = EntityEncoder(input_dim=worker_feat_dim, embed_dim=64, hidden_dim=64)
        
        # Fusion Layer (Combines Map + Command)
        self.trunk_fc = nn.Linear(128 + 64, hidden_dim)
        
        # 2. Reasoning Core (The Brain)
        self.reasoning = ReasoningCore(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # 3. Heads
        # A. Worker Selection (Pointer Network)
        self.pointer_attention = PointerAttention(query_dim=hidden_dim, embed_dim=64, use_sigmoid=False)
        
        # B. Placement Head (Spatial Offset)
        # Outputs a probability map over the 16x16 local grid
        self.placement_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, map_resolution * map_resolution) # Flattened spatial logits
        )
        
    def forward(self, 
                worker_features: torch.Tensor,     # [B, N, 5]
                command_type: torch.Tensor,        # [B, 1] (UnitTypeId)
                command_target: torch.Tensor,      # [B, 2] (Global x,y)
                local_map: torch.Tensor,           # [B, C, H, W]
                hidden_state: torch.Tensor,        # [B, Hidden]
                worker_mask: torch.Tensor = None   # [B, N]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = worker_features.size(0)
        
        # --- Encoding ---
        # 1. Command
        cmd_emb = self.cmd_embedding(command_type.squeeze(-1)) # [B, 32]
        cmd_vec = torch.cat([cmd_emb, command_target], dim=1)  # [B, 34]
        cmd_feat = self.cmd_fc(cmd_vec) # [B, 64]
        
        # 2. Map
        map_feat = self.map_cnn(local_map) # [B, 128]
        
        # 3. Combine to create 'Problem Context'
        context = torch.cat([map_feat, cmd_feat], dim=1) # [B, 192]
        trunk = F.relu(self.trunk_fc(context)) # [B, 128]
        
        # --- Reasoning ---
        # "Think" about the context (Constraint Propagation for Walling)
        new_hidden = self.reasoning(trunk, hidden_state, thinking_steps=3)
        
        # --- Heads ---
        # 1. Worker Selection
        # Encode workers first
        worker_embs = self.worker_encoder(worker_features, worker_mask) # [B, N, 64]
        # Query workers with the reasoned state
        selection_probs = self.pointer_attention(query=new_hidden, keys=worker_embs, mask=worker_mask)
        
        # 2. Placement
        placement_logits = self.placement_head(new_hidden) # [B, 256]
        
        return selection_probs, placement_logits, new_hidden
