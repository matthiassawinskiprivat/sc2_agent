"""
StarCraft 2 Unit Control Agent
Includes:
1. Observation Caching
2. Action Masking (with Dilation)
3. Coordinate Conversion
"""

import numpy as np
import cv2
from typing import Any, Tuple, Dict

class UnitControlAgent:
    def __init__(self, cnn_input_size: Tuple[int, int] = (128, 128), dilation_size: int = 3):
        self.cnn_width, self.cnn_height = cnn_input_size
        self.dilation_size = dilation_size
        self.transformed_observation = None
        self.last_game_info = None
        self.last_action = None
        self.frame_count = 0
        self.avg_processing_time = 0.0

        self._range_x = np.arange(self.cnn_width)
        self._range_y = np.arange(self.cnn_height)

    def process_transformed_observation(self, transformed_observation: np.ndarray,
                                      game_info: Any = None) -> np.ndarray:
        self.transformed_observation = transformed_observation
        if game_info:
            self.last_game_info = game_info
        return transformed_observation

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask using Inverse Mapping.
        True = Valid Action (Walkable), False = Invalid (Wall/Void).
        """
        if self.last_game_info is None:
            return np.ones(self.cnn_width * self.cnn_height, dtype=bool)

        # 1. Get Geometry
        grid = self.last_game_info.pathing_grid
        crop_x, crop_y = 0, 0
        mw, mh = grid.width, grid.height

        if hasattr(self.last_game_info, 'playable_area'):
            area = self.last_game_info.playable_area
            crop_x, crop_y = int(area.x), int(area.y)
            mw, mh = int(area.width), int(area.height)

        # 2. Scale & Offset
        scale = min(self.cnn_width / max(mw, 1), self.cnn_height / max(mh, 1))
        offset_x = max(0, (self.cnn_width - int(mw * scale)) // 2)
        offset_y = max(0, (self.cnn_height - int(mh * scale)) // 2)

        # 3. Get Raw Data
        raw_data = grid.data_numpy if hasattr(grid, "data_numpy") else np.ones((mw, mh), dtype=np.uint8)
        dim0, dim1 = raw_data.shape
        total_w = self.last_game_info.map_size.width
        total_h = self.last_game_info.map_size.height

        # 4. Inverse Map (Pixel -> Grid)
        sc2_x_1d = ((self._range_x - offset_x) / scale) + crop_x
        y_term = (self.cnn_height - 1) - self._range_y
        sc2_y_1d = ((y_term - offset_y) / scale) + crop_y

        sc2_x_grid, sc2_y_grid = np.meshgrid(sc2_x_1d.astype(int), sc2_y_1d.astype(int))

        # 5. Sample
        valid_coords = (sc2_x_grid >= 0) & (sc2_x_grid < total_w) & \
                       (sc2_y_grid >= 0) & (sc2_y_grid < total_h)

        mask_2d = np.zeros((self.cnn_height, self.cnn_width), dtype=np.uint8)
        valid_x = sc2_x_grid[valid_coords]
        valid_y = sc2_y_grid[valid_coords]

        if dim0 == total_w and dim1 == total_h:
            mask_2d[valid_coords] = (raw_data[valid_y, valid_x] != 0).astype(np.uint8)
        elif dim0 == total_h and dim1 == total_w:
            mask_2d[valid_coords] = (raw_data[valid_y, valid_x] != 0).astype(np.uint8)
        else:
            cx = np.clip(valid_x, 0, dim1 - 1)
            cy = np.clip(valid_y, 0, dim0 - 1)
            mask_2d[valid_coords] = (raw_data[cy, cx] != 0).astype(np.uint8)

        # Dilation to prevent getting stuck
        if self.dilation_size > 0:
            kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
            mask_2d = cv2.dilate(mask_2d, kernel, iterations=1)

        return mask_2d.flatten().astype(bool)

    def convert_spatial_action_to_world(self, action_idx: int) -> Tuple[float, float]:
        self.last_action = action_idx
        if self.last_game_info is None: return (0.0, 0.0)

        y_cnn, x_cnn = divmod(action_idx, self.cnn_width)

        crop_x, crop_y = 0, 0
        mw, mh = self.last_game_info.map_size.width, self.last_game_info.map_size.height

        if hasattr(self.last_game_info, 'playable_area'):
            area = self.last_game_info.playable_area
            crop_x, crop_y = area.x, area.y
            mw, mh = area.width, area.height

        scale = min(self.cnn_width / max(mw, 1), self.cnn_height / max(mh, 1))
        offset_x = max(0, (self.cnn_width - int(mw * scale)) // 2)
        offset_y = max(0, (self.cnn_height - int(mh * scale)) // 2)

        rel_x = (x_cnn - offset_x) / scale
        y_term = (self.cnn_height - 1) - y_cnn
        rel_y = (y_term - offset_y) / scale

        world_x = rel_x + crop_x
        world_y = rel_y + crop_y

        world_x = max(0, min(world_x, self.last_game_info.map_size.width))
        world_y = max(0, min(world_y, self.last_game_info.map_size.height))

        return world_x, world_y

    def reset(self):
        self.transformed_observation = None
        self.last_game_info = None
        self.last_action = None