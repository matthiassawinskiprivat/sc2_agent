"""
UI Transformer for rendering observation layers to display
Supports the full 40-Layer Architecture
"""

import numpy as np
import cv2
from typing import List, Tuple
from src.utils.layer_constants import LayerConstants

class RealTimeUITransformer:
    def __init__(self, display_size: Tuple[int, int] = (800, 600)):
        self.display_size = display_size

        # Colors are defined in BGR format (Blue, Green, Red) range 0.0-1.0
        self.layer_colors = {
            # --- GROUP 1: ALLIED STATE ---
            LayerConstants.ALLIED_UNIT_DENSITY: [1.0, 1.0, 1.0], # White
            LayerConstants.ALLIED_UNIT_ARMOR:   [1.0, 1.0, 0.0], # Cyan
            LayerConstants.ALLIED_UNIT_TYPE:    [0.0, 1.0, 0.0], # Green (Bio/Mech)
            LayerConstants.ALLIED_UNIT_SPECIAL: [0.0, 0.5, 1.0], # Orange (Massive/Psi)
            LayerConstants.ALLIED_UNIT_FLYING:  [1.0, 0.8, 0.8], # Pale Blue
            LayerConstants.ALLIED_UNIT_CLOAK:   [0.5, 0.0, 0.5], # Dark Purple

            # --- GROUP 2: ENEMY STATE ---
            LayerConstants.ENEMY_UNIT_DENSITY:  [0.5, 0.5, 0.5], # Gray
            LayerConstants.ENEMY_UNIT_ARMOR:    [1.0, 0.0, 1.0], # Magenta
            LayerConstants.ENEMY_UNIT_TYPE:     [0.2, 0.2, 0.8], # Dark Red
            LayerConstants.ENEMY_UNIT_SPECIAL:  [0.0, 0.5, 1.0], # Orange
            LayerConstants.ENEMY_UNIT_FLYING:   [0.5, 0.5, 1.0], # Pale Red
            LayerConstants.ENEMY_UNIT_CLOAK:    [0.0, 0.0, 0.8], # Deep Red

            # --- GROUP 3: VITALS ---
            LayerConstants.ALLIED_HEALTH:       [0.0, 1.0, 0.0], # Green
            LayerConstants.ENEMY_HEALTH:        [0.0, 0.0, 1.0], # Red
            LayerConstants.ALLIED_ENERGY:       [1.0, 0.0, 1.0], # Neon Purple
            LayerConstants.ENEMY_ENERGY:        [0.8, 0.5, 1.0], # Pink

            # --- GROUP 4: ALLIED OFFENSE (Blue/Cyan/Purple Themes) ---
            # Ground Targets (Blue spectrum)
            LayerConstants.ALLIED_G_DPS_BASE:    [1.0, 0.0, 0.0], # Pure Blue
            LayerConstants.ALLIED_G_DPS_LIGHT:   [1.0, 1.0, 0.0], # Cyan
            LayerConstants.ALLIED_G_DPS_ARMOR:   [0.5, 0.5, 0.0], # Teal
            LayerConstants.ALLIED_G_DPS_MASSIVE: [0.5, 0.0, 0.0], # Navy

            # Air Targets (Purple spectrum)
            LayerConstants.ALLIED_A_DPS_BASE:    [0.8, 0.0, 0.5], # Purple
            LayerConstants.ALLIED_A_DPS_LIGHT:   [1.0, 0.6, 0.8], # Lavender
            LayerConstants.ALLIED_A_DPS_ARMOR:   [0.5, 0.0, 0.3], # Indigo
            LayerConstants.ALLIED_A_DPS_MASSIVE: [0.4, 0.0, 0.2], # Dark Violet

            # --- GROUP 5: ENEMY THREAT (Red/Orange/Magenta Themes) ---
            # Ground Threats (Red/Orange spectrum)
            LayerConstants.ENEMY_G_DPS_BASE:     [0.0, 0.0, 1.0], # Pure Red
            LayerConstants.ENEMY_G_DPS_LIGHT:    [0.0, 0.6, 1.0], # Orange
            LayerConstants.ENEMY_G_DPS_ARMOR:    [0.0, 1.0, 1.0], # Yellow
            LayerConstants.ENEMY_G_DPS_MASSIVE:  [0.0, 0.0, 0.5], # Dark Red

            # Air Threats (Pink/Magenta spectrum)
            LayerConstants.ENEMY_A_DPS_BASE:     [0.6, 0.0, 1.0], # Magenta
            LayerConstants.ENEMY_A_DPS_LIGHT:    [0.8, 0.6, 1.0], # Pink
            LayerConstants.ENEMY_A_DPS_ARMOR:    [0.4, 0.0, 0.8], # Deep Pink
            LayerConstants.ENEMY_A_DPS_MASSIVE:  [0.3, 0.0, 0.5], # Dark Magenta

            # --- GROUP 6: UTILITY ---
            LayerConstants.ALLIED_HPS_FIELD:     [0.5, 1.0, 0.0], # Bright Green
            LayerConstants.ENEMY_HPS_FIELD:      [0.0, 0.5, 0.0], # Dark Green

            # --- GROUP 7: INTEL ---
            LayerConstants.ALLIED_DETECTION:     [0.0, 1.0, 1.0], # Yellow
            LayerConstants.ENEMY_DETECTION:      [0.0, 0.5, 1.0], # Orange
            LayerConstants.TERRAIN_PATHABLE:     [0.2, 0.2, 0.2], # Dark Gray
            LayerConstants.TERRAIN_CREEP:        [0.8, 0.0, 0.8], # Bright Purple

            # --- GROUP 8: PROTOCOL ---
            LayerConstants.COMMAND_ASSIGNMENT:   [0.0, 1.0, 0.0], # Green Overlay
            LayerConstants.COMMAND_OBJECTIVE:    [0.0, 1.0, 1.0], # Yellow Overlay
        }

    def update_display(self, observation: np.ndarray, enabled_layers: List[str],
                      fps: float = 0) -> np.ndarray:
        if observation is None:
            return self._create_black_frame()

        num_layers, obs_height, obs_width = observation.shape
        frame = np.zeros((obs_height, obs_width, 3), dtype=np.float32)

        # Blend layers
        for layer_name in enabled_layers:
            if layer_name not in LayerConstants.LAYERS: continue

            layer_idx = LayerConstants.LAYERS[layer_name]
            if layer_idx >= num_layers: continue

            layer_data = observation[layer_idx]
            if layer_data.max() == 0: continue

            # Data is already normalized (0-1) from transformer
            # We just multiply by color
            color = self.layer_colors.get(layer_name, [1, 1, 1])

            # Additive Blending: Max(current, new) ensures layers don't wash out
            for c in range(3):
                frame[:, :, c] = np.maximum(frame[:, :, c], layer_data * color[c])

        # Clip and Convert
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_NEAREST)

        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    def _create_black_frame(self) -> np.ndarray:
        return np.zeros((*self.display_size[::-1], 3), dtype=np.uint8)