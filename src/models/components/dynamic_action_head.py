import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class DynamicActionHead(nn.Module):
    """
    A flexible Action Head module that can grow new output heads based on configuration.
    
    Structure:
    1. Selector Head: Linear(Input -> N_Heads). Decides WHICH head to use (e.g., Move vs Load).
    2. Spatial Heads: Dict[Name -> Linear(Input -> Spatial_Grid)]. One grid per action type.
    """
    def __init__(self, input_dim: int, head_names: List[str], grid_size: int = 128*128):
        super().__init__()
        self.head_names = head_names
        self.grid_size = grid_size
        
        # 1. Selector Head (Logits for Softmax)
        self.selector = nn.Linear(input_dim, len(head_names))
        
        # 2. Spatial Heads
        # We use ModuleDict so PyTorch registers sub-modules properly
        self.spatial_heads = nn.ModuleDict()
        
        for name in head_names:
            # Each head outputs a flattened spatial logit map
            self.spatial_heads[name] = nn.Linear(input_dim, grid_size)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [Batch, Input_Dim] - The context vector (Reasoned State).
            
        Returns:
            selector_logits: [Batch, N_Heads]
            spatial_logits: Dict[HeadName -> [Batch, Grid_Size]]
        """
        selector_logits = self.selector(x)
        
        spatial_outputs = {}
        for name, head_module in self.spatial_heads.items():
            spatial_outputs[name] = head_module(x)
            
        return selector_logits, spatial_outputs
