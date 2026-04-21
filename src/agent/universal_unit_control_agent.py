"""
StarCraft 2 Universal Unit Control Agent
Uses a shared 16x16 spatial head for all commands.
The meaning of the spatial index changes based on the chosen command.
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional, List
from sc2.position import Point2
from src.agent.unit_control_agent import UnitControlAgent
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

class UniversalUnitControlAgent(UnitControlAgent):
    def __init__(self, cnn_input_size: Tuple[int, int] = (128, 128), 
                 grid_size: int = 128, 
                 dilation_size: int = 3):
        # We keep grid_size for the observation layers (CNN)
        self.spatial_resolution = 32
        super().__init__(cnn_input_size=cnn_input_size, grid_size=grid_size, dilation_size=dilation_size)

    def get_action_mask(self, head_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Universal pathing mask (spatial only) when head_names is None.

        When head_names is set (tests / tooling), returns a flat mask of size
        FIXED_UNIVERSAL_COMMAND_DIM + spatial_resolution^2 with the command prefix
        True only for indices < len(head_names) (matches env / policy layout).
        """
        n_spatial = self.spatial_resolution ** 2
        spatial_mask = np.ones(n_spatial, dtype=bool)

        if head_names is None:
            return spatial_mask

        cmd = np.zeros(FIXED_UNIVERSAL_COMMAND_DIM, dtype=bool)
        cmd[: len(head_names)] = True
        return np.concatenate([cmd, spatial_mask])

    def convert_spatial_action_to_world(self, command_name: str, spatial_idx: int, active_unit: Any = None) -> Tuple[float, float]:
        """
        Interprets the spatial resolution grid based on the command.
        """
        if active_unit is None: return (0.0, 0.0)
        
        # 1. Case A: LOCAL Micro (Centered on unit)
        if "_local" in command_name:
            res = self.spatial_resolution
            gy, gx = divmod(spatial_idx, res)
            
            # ObservationTransformer draws unit at px=16, py=15 in 32x32 tactical
            center_x = 16
            center_y = 15
            
            world_dx = gx - center_x
            world_dy = center_y - gy
            
            world_x = active_unit.position.x + world_dx
            world_y = active_unit.position.y + world_dy

            # Clamp to playable bounds (preferred) to avoid targeting void.
            if self.last_game_info:
                eps = 1e-3
                if hasattr(self.last_game_info, "playable_area") and self.last_game_info.playable_area is not None:
                    area = self.last_game_info.playable_area
                    min_x = float(area.x)
                    min_y = float(area.y)
                    max_x = float(area.x + area.width) - eps
                    max_y = float(area.y + area.height) - eps
                else:
                    min_x = 0.0
                    min_y = 0.0
                    max_x = float(self.last_game_info.map_size.width) - eps
                    max_y = float(self.last_game_info.map_size.height) - eps

                world_x = max(min_x, min(float(world_x), max_x))
                world_y = max(min_y, min(float(world_y), max_y))
            return float(world_x), float(world_y)
            
        # 2. Case B: GLOBAL Navigation (Whole map)
        # Use robust base class logic (handles non-128x128 maps)
        # Ensure we use 32x32 grid decoding for the spatial_idx
        old_grid_size = self.grid_size
        self.grid_size = self.spatial_resolution
        try:
            return super().convert_spatial_action_to_world(spatial_idx, active_unit=active_unit)
        finally:
            self.grid_size = old_grid_size
