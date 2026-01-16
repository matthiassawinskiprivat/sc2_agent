import torch
import torch.nn as nn

class ReasoningCore(nn.Module):
    """
    Implements the 'System 2' Reasoning Loop.
    Instead of passing state through time (standard RNN), this passes state through
    'thought depth' within a single forward pass.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # We use GRUCell because we manually control the loop
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        
    def forward(self, input_tensor: torch.Tensor, hidden_state: torch.Tensor, thinking_steps: int = 3) -> torch.Tensor:
        """
        Args:
            input_tensor: [Batch, Input_Dim] - The current observation/context.
            hidden_state: [Batch, Hidden_Dim] - The memory from previous Game Step.
            thinking_steps: How many times to loop the reasoning core.
            
        Returns:
            new_hidden_state: [Batch, Hidden_Dim] - The final state after thinking.
        """
        current_state = hidden_state
        
        # The Reasoning Loop
        # In the first iteration, we combine the 'Input' with the 'Old Memory'.
        # In subsequent iterations, we keep feeding the SAME 'Input', but the
        # 'Current State' evolves, allowing the network to refine its understanding.
        for _ in range(thinking_steps):
            current_state = self.gru_cell(input_tensor, current_state)
            
        return current_state
