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
    def __init__(self, cnn_input_size: Tuple[int, int] = (128, 128), grid_size: int = 128, dilation_size: int = 3):
        self.cnn_width, self.cnn_height = cnn_input_size
        self.grid_size = grid_size
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
            return np.ones(self.grid_size * self.grid_size, dtype=bool)

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

        # Downsample the 128x128 mask to grid_size x grid_size
        resized_mask = cv2.resize(mask_2d, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)

        return (resized_mask.flatten() > 0).astype(bool)


    def get_transport_mask(self, units: list, game_info: Any) -> np.ndarray:
        """
        Generates a mask for the load action, highlighting transport units.
        """
        if not game_info:
            return np.ones(self.grid_size * self.grid_size, dtype=bool)

        # 1. Geometry Setup (Same as get_action_mask)
        grid = game_info.pathing_grid
        crop_x, crop_y = 0, 0
        mw, mh = grid.width, grid.height

        if hasattr(game_info, 'playable_area'):
            area = game_info.playable_area
            crop_x, crop_y = int(area.x), int(area.y)
            mw, mh = int(area.width), int(area.height)

        scale = min(self.cnn_width / max(mw, 1), self.cnn_height / max(mh, 1))
        offset_x = max(0, (self.cnn_width - int(mw * scale)) // 2)
        offset_y = max(0, (self.cnn_height - int(mh * scale)) // 2)

        # 2. Create Blank Mask
        mask_2d = np.zeros((self.cnn_height, self.cnn_width), dtype=np.uint8)

        # 3. Mark Transports
        # We define transport types here to avoid hard dependency on sc2.ids if possible,
        # but for now we assume the caller passes units with type_id.
        # Common transport IDs (integers):
        # MEDIVAC: 54, OVERLORD: 106, WARPPRISM: 81, WARPPRISMFLYING: 85
        # We can also check by name if passed object supports it, or import UnitTypeId.
        # Since this is part of the agent, let's try to match by ID values or import if available.
        # To be safe and clean, we will assume standard sc2 Unit objects.
        
        # UnitTypeId values
        TRANSPORT_IDS = {54, 106, 81, 136, 85} # Medivac, Overlord, WarpPrism, WarpPrismPhasing, etc.
        # Or better, check specific known IDs.
        # Medivac=54, Overlord=106, OverlordTransport=893? 
        # Let's trust the passed unit.type_id.
        # We'll use a broader set or rely on the caller to filter?
        # The spec says "Transport Identification: The mask must identify all allied transport units".
        # So we should filter here.
        
        from sc2.ids.unit_typeid import UnitTypeId
        valid_types = {
            UnitTypeId.MEDIVAC, 
            UnitTypeId.OVERLORD, # Standard overlord (with upgrade) or OverlordTransport?
            UnitTypeId.OVERLORDTRANSPORT,
            UnitTypeId.WARPPRISM
        }

        for unit in units:
            if unit.type_id in valid_types:
                # World -> CNN
                # rel_x = unit.position.x - crop_x
                # x_cnn = rel_x * scale + offset_x
                
                rel_x = unit.position.x - crop_x
                rel_y = unit.position.y - crop_y
                
                # Y is flipped in SC2 grid data usually, but unit.position is standard cartesian?
                # In get_action_mask: 
                # sc2_y_1d = ((y_term - offset_y) / scale) + crop_y
                # => y_term = (sc2_y - crop_y) * scale + offset_y
                # => (self.cnn_height - 1) - y_range = ...
                # => y_range = (self.cnn_height - 1) - ((sc2_y - crop_y) * scale + offset_y)
                
                x_cnn = int((rel_x * scale) + offset_x)
                
                # CAUTION: Coordinate systems.
                # get_action_mask iterates Y from 0..cnn_height.
                # y_term = (self.cnn_height - 1) - self._range_y
                # sc2_y = ((y_term - offset_y) / scale) + crop_y
                # Inverting for pixel Y:
                # sc2_y - crop_y = (y_term - offset_y) / scale
                # (sc2_y - crop_y) * scale = y_term - offset_y
                # (sc2_y - crop_y) * scale + offset_y = y_term
                # (sc2_y - crop_y) * scale + offset_y = (self.cnn_height - 1) - pixel_y
                # pixel_y = (self.cnn_height - 1) - ((sc2_y - crop_y) * scale + offset_y)
                
                y_cnn = int((self.cnn_height - 1) - ((rel_y * scale) + offset_y))
                
                if 0 <= x_cnn < self.cnn_width and 0 <= y_cnn < self.cnn_height:
                    mask_2d[y_cnn, x_cnn] = 255

        # 4. Dilation
        # We use a slightly larger dilation for load to ensure the "click" hits the unit radius
        if self.dilation_size > 0:
            # Maybe +1 to standard dilation for ease of use
            kernel_size = self.dilation_size + 2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_2d = cv2.dilate(mask_2d, kernel, iterations=1)

        # 5. Resize to Action Grid
        resized_mask = cv2.resize(mask_2d, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
        
        return (resized_mask.flatten() > 0).astype(bool)

    def get_unload_mask(self, units: list, game_info: Any) -> np.ndarray:
        """
        Generates a mask for the unload action.
        1. Excludes unpathable terrain with a 2-pixel safety margin (erosion).
        2. Excludes buildings and resources with exact footprints.
        """
        if not game_info:
            return np.ones(self.grid_size * self.grid_size, dtype=bool)

        # 1. Geometry Setup (Same as get_action_mask)
        grid = game_info.pathing_grid
        crop_x, crop_y = 0, 0
        mw, mh = grid.width, grid.height

        if hasattr(game_info, 'playable_area'):
            area = game_info.playable_area
            crop_x, crop_y = int(area.x), int(area.y)
            mw, mh = int(area.width), int(area.height)

        scale = min(self.cnn_width / max(mw, 1), self.cnn_height / max(mh, 1))
        offset_x = max(0, (self.cnn_width - int(mw * scale)) // 2)
        offset_y = max(0, (self.cnn_height - int(mh * scale)) // 2)

        # 2. Get Raw Pathing Data
        raw_data = grid.data_numpy if hasattr(grid, "data_numpy") else np.ones((mw, mh), dtype=np.uint8)
        total_w = game_info.map_size.width
        total_h = game_info.map_size.height

        # 3. Create Terrain-Only Mask (Walkable = 255)
        # Use Inverse Mapping to get pathing into CNN space
        sc2_x_1d = ((self._range_x - offset_x) / scale) + crop_x
        y_term = (self.cnn_height - 1) - self._range_y
        sc2_y_1d = ((y_term - offset_y) / scale) + crop_y

        sc2_x_grid, sc2_y_grid = np.meshgrid(sc2_x_1d.astype(int), sc2_y_1d.astype(int))
        valid_coords = (sc2_x_grid >= 0) & (sc2_x_grid < total_w) & \
                       (sc2_y_grid >= 0) & (sc2_y_grid < total_h)

        terrain_mask = np.zeros((self.cnn_height, self.cnn_width), dtype=np.uint8)
        valid_x = sc2_x_grid[valid_coords]
        valid_y = sc2_y_grid[valid_coords]

        # Sample from raw_data
        dim0, dim1 = raw_data.shape
        if dim0 == total_w and dim1 == total_h:
            terrain_mask[valid_coords] = (raw_data[valid_y, valid_x] != 0).astype(np.uint8) * 255
        elif dim0 == total_h and dim1 == total_w:
            terrain_mask[valid_coords] = (raw_data[valid_y, valid_x] != 0).astype(np.uint8) * 255
        else:
            cx = np.clip(valid_x, 0, dim1 - 1)
            cy = np.clip(valid_y, 0, dim0 - 1)
            terrain_mask[valid_coords] = (raw_data[cy, cx] != 0).astype(np.uint8) * 255

        # 4. Erode Terrain Mask (Safety Margin from Cliffs)
        # 2-pixel erosion to ensure units aren't dropped on the edge
        kernel = np.ones((5, 5), np.uint8) # 5x5 kernel approx 2px radius
        eroded_terrain = cv2.erode(terrain_mask, kernel, iterations=1)

        # 5. Create Entity Footprint Mask (Blocked = 255)
        entity_mask = np.zeros((self.cnn_height, self.cnn_width), dtype=np.uint8)
        
        # Filter for buildings and resources
        # Resources: MINERALFIELD, VESPENEGEYSER
        for unit in units:
            if unit.is_structure or unit.is_mineral_field or unit.is_vespene_geyser:
                rel_x = unit.position.x - crop_x
                rel_y = unit.position.y - crop_y
                
                x_cnn = int((rel_x * scale) + offset_x)
                y_cnn = int((self.cnn_height - 1) - ((rel_y * scale) + offset_y))
                
                if 0 <= x_cnn < self.cnn_width and 0 <= y_cnn < self.cnn_height:
                    # Use footprint_radius if available, else radius
                    radius = getattr(unit, "footprint_radius", unit.radius)
                    r_px = int(radius * scale)
                    if r_px < 1: r_px = 1
                    cv2.circle(entity_mask, (x_cnn, y_cnn), r_px, 255, -1)

        # 6. Final Combine
        # Valid = (Eroded Terrain) AND (NOT Entity)
        final_mask_2d = cv2.bitwise_and(eroded_terrain, cv2.bitwise_not(entity_mask))

        # 7. Resize to Action Grid
        resized_mask = cv2.resize(final_mask_2d, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
        
        return (resized_mask.flatten() > 0).astype(bool)

    def convert_spatial_action_to_world(self, action_idx: int) -> Tuple[float, float]:
        self.last_action = action_idx
        if self.last_game_info is None: return (0.0, 0.0)

        # Convert 1D action index from grid (e.g., 16x16) to 2D grid coordinates
        grid_y, grid_x = divmod(action_idx, self.grid_size)

        # Scale grid coordinates to CNN coordinates (e.g., 128x128)
        # This targets the center of the grid cell
        scale_factor = self.cnn_width / self.grid_size
        y_cnn = (grid_y * scale_factor) + (scale_factor / 2)
        x_cnn = (grid_x * scale_factor) + (scale_factor / 2)

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