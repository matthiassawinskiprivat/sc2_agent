"""
StarCraft 2 Unit Control Agent
Includes:
1. Observation Caching
2. Action Masking (aligned with Visual Transformer)
3. Coordinate Conversion (Image Index -> SC2 World Point)
"""

import numpy as np
import cv2
from typing import Any, Tuple, Dict

class UnitControlAgent:
    def __init__(self, cnn_input_size: Tuple[int, int] = (128, 128)):
        self.cnn_width, self.cnn_height = cnn_input_size
        self.transformed_observation = None
        self.last_game_info = None
        self.frame_count = 0
        self.avg_processing_time = 0.0

    def process_transformed_observation(self, transformed_observation: np.ndarray,
                                      game_info: Any = None) -> np.ndarray:
        """
        Store the latest observation and game context.
        This is called by the Hierarchical Agent System before the agent acts.
        """
        self.transformed_observation = transformed_observation
        if game_info:
            self.last_game_info = game_info
        return transformed_observation

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for the action space.
        True = Valid Action (Walkable), False = Invalid (Wall/Void).
        Matches the visual transformation logic: Transpose -> Resize -> Place -> FlipUD.
        """
        # If we haven't seen the map yet, allow everything (safe fallback)
        if self.last_game_info is None:
            return np.ones(self.cnn_width * self.cnn_height, dtype=bool)

        # 1. Get Raw Grid [Width, Height]
        raw_grid = self.last_game_info.pathing_grid.data_numpy.astype(np.uint8)
        map_w, map_h = raw_grid.shape

        # 2. Transpose to [Height, Width] for Image operations
        grid_img = raw_grid.T

        # 3. Calculate Scale & Offsets (Letterboxing)
        scale = min(self.cnn_width / map_w, self.cnn_height / map_h)
        new_w = int(map_w * scale)
        new_h = int(map_h * scale)

        offset_x = int((self.cnn_width - new_w) / 2)
        offset_y = int((self.cnn_height - new_h) / 2)

        # 4. Resize the valid pathing area
        # Use Nearest Neighbor to keep walls solid (0 or 1)
        resized = cv2.resize(grid_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 5. Place into Canvas
        # Create black canvas (all invalid)
        mask_canvas = np.zeros((self.cnn_height, self.cnn_width), dtype=np.uint8)

        # Paste the resized map into the center (handling slight rounding errors safely)
        y_end = min(offset_y + new_h, self.cnn_height)
        x_end = min(offset_x + new_w, self.cnn_width)

        # Safe slice insertion
        mask_canvas[offset_y:y_end, offset_x:x_end] = resized[:(y_end-offset_y), :(x_end-offset_x)]

        # 6. Flip Vertically
        # SC2 (0,0) is Bottom-Left. Image (0,0) is Top-Left.
        final_mask = np.flipud(mask_canvas)

        return final_mask.flatten().astype(bool)

    def convert_spatial_action_to_world(self, action_idx: int) -> Tuple[float, float]:
        """
        Converts a flat neural network action index (0..16383) to SC2 World Coordinates.
        Performs the exact inverse of the ObservationTransformer logic.
        """
        if self.last_game_info is None:
            return (0.0, 0.0)

        # 1. Decode Index to Image Coordinates
        # Row (y), Col (x)
        y_cnn, x_cnn = divmod(action_idx, self.cnn_width)

        map_w = self.last_game_info.map_size.width
        map_h = self.last_game_info.map_size.height

        scale = min(self.cnn_width / map_w, self.cnn_height / map_h)
        offset_x = int((self.cnn_width - int(map_w * scale)) / 2)
        offset_y = int((self.cnn_height - int(map_h * scale)) / 2)

        # 2. Inverse X Transform
        # Forward: img_x = sc2_x * scale + offset_x
        # Inverse: sc2_x = (img_x - offset_x) / scale
        world_x = (x_cnn - offset_x) / scale

        # 3. Inverse Y Transform
        # Forward: img_y = CanvasH - 1 - (sc2_y * scale + offset_y)
        # Inverse:
        #   img_y - (CanvasH - 1) = -(sc2_y * scale + offset_y)
        #   (CanvasH - 1) - img_y = sc2_y * scale + offset_y
        #   ((CanvasH - 1 - img_y) - offset_y) / scale = sc2_y

        y_term = (self.cnn_height - 1) - y_cnn
        world_y = (y_term - offset_y) / scale

        return world_x, world_y

    def reset(self):
        self.transformed_observation = None
        self.last_game_info = None