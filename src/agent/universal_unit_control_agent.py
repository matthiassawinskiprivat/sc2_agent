"""
StarCraft 2 Universal Unit Control Agent
Uses a shared 16x16 spatial head for all commands.
The meaning of the spatial index changes based on the chosen command.
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional, List
from sc2.position import Point2
from src.agent.unit_control_agent import UnitControlAgent

class UniversalUnitControlAgent(UnitControlAgent):
    def __init__(self, cnn_input_size: Tuple[int, int] = (128, 128), 
                 grid_size: int = 128, 
                 dilation_size: int = 3,
                 spatial_resolution: int = 16):
        # We keep grid_size for the observation layers (CNN)
        super().__init__(cnn_input_size=cnn_input_size, grid_size=grid_size, dilation_size=dilation_size)
        self.spatial_resolution = spatial_resolution # Default 16 (for 16x16 grid)

    def get_action_mask(self, head_names: List[str]) -> np.ndarray:
        """
        Universal Masking: 
        1. Command Selector Mask (Always all True for now)
        2. Shared Spatial Mask (Always all True for speed)
        Returns a flat mask of size: len(head_names) + spatial_resolution^2
        """
        n_cmds = len(head_names)
        n_spatial = self.spatial_resolution ** 2
        return np.ones(n_cmds + n_spatial, dtype=bool)

    def convert_spatial_action_to_world(self, command_name: str, spatial_idx: int, active_unit: Any = None) -> Tuple[float, float]:
        """
        The Core Logic: Interprets the 16x16 grid based on the command.
        """
        if active_unit is None: return (0.0, 0.0)
        
        # 1. Decode 1D index to 2D 16x16 coordinates
        # Grid is 16x16, so row = idx // 16, col = idx % 16
        res = self.spatial_resolution
        gy, gx = divmod(spatial_idx, res)
        
        # 2. Case A: LOCAL Micro (Centered on unit)
        # Each pixel represents 1.0 world units.
        if "_local" in command_name:
            center = res // 2
            world_dx = gx - center
            world_dy = center - gy # Inverted Y
            
            world_x = active_unit.position.x + world_dx
            world_y = active_unit.position.y + world_dy
            
        # 3. Case B: GLOBAL Navigation (Whole map)
        # Map is 128x128, Grid is 16x16. Each pixel = 8x8 world units.
        else:
            scale = 128 / res # e.g. 8.0
            # Target center of the 8x8 block
            world_x = (gx * scale) + (scale / 2)
            world_y = (127 - (gy * scale)) - (scale / 2) # Inverted Y

        # 4. Final Clamp
        if self.last_game_info:
            world_x = max(0, min(world_x, self.last_game_info.map_size.width))
            world_y = max(0, min(world_y, self.last_game_info.map_size.height))
            
        return float(world_x), float(world_y)
