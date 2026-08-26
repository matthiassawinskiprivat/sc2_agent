"""
UI Transformer for rendering observation layers to display
Supports the 46-Layer Architecture with Orthogonal Attributes
"""

import re
import numpy as np
import cv2
from typing import List, Tuple
from src.utils.layer_constants import LayerConstants

# Command protocol layers render on top of unit density/attribute layers so they
# stay visible when np.maximum would let white density wash out green assignment.
_PROTOCOL_OVERLAY_LAYERS = frozenset({
    LayerConstants.COMMAND_ASSIGNMENT,
    LayerConstants.COMMAND_OBJECTIVE,
})

# Typed DPS channels store bonus-only when the matching *_dps_base is present.
# For debug display, reconstruct total vs-attribute threat (base + bonus).
_TYPED_DPS_LAYER_RE = re.compile(
    r"^(allied|enemy)_(g|a)_dps_(light|armor|massive)$"
)


class RealTimeUITransformer:
    def __init__(self, display_size: Tuple[int, int] = (600, 600)):
        self.display_size = display_size

        self.layer_colors = {
            # --- GROUP 1: ALLIED STATE ---
            LayerConstants.ALLIED_UNIT_DENSITY:     [1.0, 1.0, 1.0], # White
            LayerConstants.ALLIED_UNIT_BIOLOGICAL:  [0.0, 1.0, 0.0], # Green
            LayerConstants.ALLIED_UNIT_MECHANICAL:  [0.0, 0.0, 1.0], # Blue
            LayerConstants.ALLIED_UNIT_ARMORED:     [1.0, 1.0, 0.0], # Yellow (Bio/Mech + Armored = Mix)
            LayerConstants.ALLIED_UNIT_MASSIVE:     [1.0, 0.5, 0.0], # Orange
            LayerConstants.ALLIED_UNIT_FLYING:      [0.0, 1.0, 1.0], # Cyan
            LayerConstants.ALLIED_UNIT_CLOAK:       [0.5, 0.0, 0.5], # Purple

            # --- GROUP 2: ENEMY STATE ---
            LayerConstants.ENEMY_UNIT_DENSITY:      [0.5, 0.5, 0.5], # Gray
            LayerConstants.ENEMY_UNIT_BIOLOGICAL:   [0.5, 0.0, 0.0], # Dark Red
            LayerConstants.ENEMY_UNIT_MECHANICAL:   [0.0, 0.0, 0.5], # Dark Blue
            LayerConstants.ENEMY_UNIT_ARMORED:      [0.5, 0.5, 0.0], # Olive
            LayerConstants.ENEMY_UNIT_MASSIVE:      [0.6, 0.2, 0.0], # Brown
            LayerConstants.ENEMY_UNIT_FLYING:       [0.5, 0.5, 1.0], # Lavender
            LayerConstants.ENEMY_UNIT_CLOAK:        [0.2, 0.0, 0.2], # Dark Purple

            # --- GROUP 3: VITALS ---
            LayerConstants.ALLIED_HEALTH:           [0.0, 1.0, 0.0],
            LayerConstants.ENEMY_HEALTH:            [1.0, 0.0, 0.0],
            LayerConstants.ALLIED_ENERGY:           [1.0, 0.0, 1.0],
            LayerConstants.ENEMY_ENERGY:            [0.8, 0.5, 1.0],

            # --- GROUP 4: ALLIED OFFENSE (Blue/Cyan Themes) ---
            LayerConstants.ALLIED_G_DPS_BASE:       [0.0, 0.0, 1.0],
            LayerConstants.ALLIED_G_DPS_LIGHT:      [0.0, 0.5, 1.0],
            LayerConstants.ALLIED_G_DPS_ARMOR:      [0.0, 1.0, 1.0],
            LayerConstants.ALLIED_G_DPS_MASSIVE:    [0.0, 0.0, 0.5],

            LayerConstants.ALLIED_A_DPS_BASE:       [0.5, 0.0, 1.0],
            LayerConstants.ALLIED_A_DPS_LIGHT:      [0.7, 0.3, 1.0],
            LayerConstants.ALLIED_A_DPS_ARMOR:      [0.3, 0.0, 0.7],
            LayerConstants.ALLIED_A_DPS_MASSIVE:    [1.0, 0.35, 1.0],  # Bright magenta-pink

            # --- GROUP 5: ENEMY THREAT (Red/Orange Themes) ---
            LayerConstants.ENEMY_G_DPS_BASE:        [1.0, 0.0, 0.0],
            LayerConstants.ENEMY_G_DPS_LIGHT:       [1.0, 0.5, 0.0],
            LayerConstants.ENEMY_G_DPS_ARMOR:       [1.0, 1.0, 0.0],
            LayerConstants.ENEMY_G_DPS_MASSIVE:     [0.5, 0.0, 0.0],

            LayerConstants.ENEMY_A_DPS_BASE:        [1.0, 0.0, 1.0],
            LayerConstants.ENEMY_A_DPS_LIGHT:       [1.0, 0.5, 0.5],
            LayerConstants.ENEMY_A_DPS_ARMOR:       [0.8, 0.0, 0.0],
            LayerConstants.ENEMY_A_DPS_MASSIVE:     [1.0, 0.2, 0.55],  # Hot pink-red

            # --- UTILITY ---
            LayerConstants.ALLIED_HPS_FIELD:        [0.5, 1.0, 0.0],
            LayerConstants.ENEMY_HPS_FIELD:         [0.0, 0.5, 0.0],

            # --- INTEL ---
            LayerConstants.ALLIED_DETECTION:        [0.0, 1.0, 1.0],
            LayerConstants.ENEMY_DETECTION:         [1.0, 0.5, 0.0],
            LayerConstants.TERRAIN_PATHABLE:        [0.3, 0.3, 0.3],
            LayerConstants.TERRAIN_CREEP:           [0.4, 0.0, 0.4],

            LayerConstants.COMMAND_ASSIGNMENT:      [1.0, 0.0, 1.0],  # Magenta (distinct from allied bio green)
            LayerConstants.COMMAND_OBJECTIVE:       [1.0, 1.0, 0.0],

            # --- SPLASH ---
            LayerConstants.ALLIED_SPLASH:           [0.2, 0.8, 1.0],  # Light cyan
            LayerConstants.ENEMY_SPLASH:            [1.0, 0.3, 0.1],  # Hot orange-red
        }

    def update_display(self, observation: np.ndarray, enabled_layers: List[str],
                       fps: float = 0, active_layers: List[str] = None) -> np.ndarray:
        if observation is None:
            return self._create_black_frame()

        num_layers, obs_height, obs_width = observation.shape
        frame = np.zeros((obs_height, obs_width, 3), dtype=np.float32)

        # Map layer names to observation indices.
        if active_layers:
            layer_map = {name: i for i, name in enumerate(active_layers)}
        else:
            layer_map = LayerConstants.LAYERS

        base_layers = [n for n in enabled_layers if n not in _PROTOCOL_OVERLAY_LAYERS]
        overlay_layers = [n for n in enabled_layers if n in _PROTOCOL_OVERLAY_LAYERS]

        for layer_name in base_layers + overlay_layers:
            if layer_name not in layer_map:
                continue
            layer_idx = layer_map[layer_name]

            if layer_idx >= num_layers:
                continue

            layer_data = observation[layer_idx]

            # Typed DPS tensors are bonus-only when base is present; show total threat in UI.
            typed = _TYPED_DPS_LAYER_RE.match(layer_name)
            if typed is not None:
                base_name = f"{typed.group(1)}_{typed.group(2)}_dps_base"
                base_idx = layer_map.get(base_name)
                if base_idx is not None and base_idx < num_layers:
                    layer_data = layer_data + observation[base_idx]

            if layer_data.max() == 0:
                continue

            visual_data = np.clip(layer_data, 0.0, 1.0)

            color = self.layer_colors.get(layer_name, [1, 1, 1])

            color_arr = np.array(color).reshape(1, 1, 3)
            data_expanded = visual_data[:, :, np.newaxis]
            colored_layer = data_expanded * color_arr

            if layer_name in _PROTOCOL_OVERLAY_LAYERS:
                mask = visual_data > 0
                frame[mask] = colored_layer[mask]
            else:
                frame = np.maximum(frame, colored_layer)

        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_NEAREST)

        return frame

    def _create_black_frame(self) -> np.ndarray:
        return np.zeros((*self.display_size[::-1], 3), dtype=np.uint8)
