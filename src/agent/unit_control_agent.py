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


    def get_transport_mask(self, units: list, game_info: Any, is_transport: bool = False, active_unit: Any = None, cast_range: float = 4.0) -> np.ndarray:
        """
        Generates a mask for the load action.
        If is_transport is True (actor is a Medivac), it returns the terrain mask (cliffs masked).
        If is_transport is False (actor is ground), it highlights allied transports with circle targets.
        
        NEW: If active_unit is provided, we mask EVERYTHING if no valid target is within interaction range.
        """
        from sc2.ids.unit_typeid import UnitTypeId
        
        TRANSPORT_TYPES = {
            UnitTypeId.MEDIVAC, 
            UnitTypeId.OVERLORD, 
            UnitTypeId.OVERLORDTRANSPORT,
            UnitTypeId.WARPPRISM,
            UnitTypeId.WARPPRISMPHASING
        }

        # --- Range Check Optimization ---
        if active_unit:
            active_radius = getattr(active_unit, "radius", 0.0)
            can_interact = False

            if is_transport:
                # Active: Transport. Target: Loadable Ground Units.
                # We assume anything with cargo_max=0 and not a structure is loadable.
                # Also exclude self.
                for u in units:
                    if u.tag == active_unit.tag: continue
                    if u.is_structure: continue
                    if u.cargo_max > 0: continue # Skip other transports
                    
                    # Calculate distance
                    # python-sc2 units have distance_to. 
                    # If we are mocking, ensure position is available.
                    dist = active_unit.distance_to(u)
                    target_radius = getattr(u, "radius", 0.0)
                    
                    # Effective Range = Cast + RadiusA + RadiusB
                    if dist <= (cast_range + active_radius + target_radius):
                        can_interact = True
                        break
            else:
                # Active: Ground. Target: Transports.
                for u in units:
                    if u.tag == active_unit.tag: continue
                    if u.type_id not in TRANSPORT_TYPES: continue
                    
                    dist = active_unit.distance_to(u)
                    target_radius = getattr(u, "radius", 0.0)
                    
                    if dist <= (cast_range + active_radius + target_radius):
                        can_interact = True
                        break
            
            # If NO unit is in range, strictly forbid the load action.
            if not can_interact:
                return np.zeros(self.grid_size * self.grid_size, dtype=bool)
        # -------------------------------

        if game_info:
            self.last_game_info = game_info

        if is_transport:
             # Task 3: Transport actor uses Terrain Mask (same as move mask)
             # This masks out cliffs/void, focusing search on valid ground.
             return self.get_action_mask()

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

        # 3. Mark Targets
        for unit in units:
            # Ground unit looks for allied transports
            if unit.type_id in TRANSPORT_TYPES:
                rel_x = unit.position.x - crop_x
                rel_y = unit.position.y - crop_y
                
                x_cnn = int((rel_x * scale) + offset_x)
                y_cnn = int((self.cnn_height - 1) - ((rel_y * scale) + offset_y))
                
                if 0 <= x_cnn < self.cnn_width and 0 <= y_cnn < self.cnn_height:
                    # Draw Circle based on radius (Task 4)
                    radius = getattr(unit, "radius", 1.0)
                    r_px = int(radius * scale)
                    if r_px < 1: r_px = 1
                    cv2.circle(mask_2d, (x_cnn, y_cnn), r_px, 255, -1)

        # 4. Resize to Action Grid
        # We skip dilation here because we already used the unit radius (circle).
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
        # 1-pixel erosion (3x3 kernel) to ensure units aren't dropped on the extreme edge
        kernel = np.ones((3, 3), np.uint8) 
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
                    # Use footprint_radius if available, else radius. Fallback to 0.0 if both None.
                    radius = getattr(unit, "footprint_radius", None)
                    if radius is None:
                        radius = getattr(unit, "radius", 0.5) # Default small radius if missing
                    
                    if radius is not None:
                        r_px = int(radius * scale)
                        if r_px < 1: r_px = 1
                        cv2.circle(entity_mask, (x_cnn, y_cnn), r_px, 255, -1)

        # 6. Final Combine
        # Valid = (Eroded Terrain) AND (NOT Entity)
        final_mask_2d = cv2.bitwise_and(eroded_terrain, cv2.bitwise_not(entity_mask))

        # 7. Resize to Action Grid
        resized_mask = cv2.resize(final_mask_2d, (self.grid_size, self.grid_size), interpolation=cv2.INTER_NEAREST)
        
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

    def get_grid_index_at_world_pos(self, pos) -> int:
        """
        Inverse of convert_spatial_action_to_world.
        Maps SC2 World coordinates back to a 1D grid index.
        """
        if self.last_game_info is None:
            return 0

        crop_x, crop_y = 0, 0
        mw, mh = self.last_game_info.map_size.width, self.last_game_info.map_size.height

        if hasattr(self.last_game_info, 'playable_area'):
            area = self.last_game_info.playable_area
            crop_x, crop_y = area.x, area.y
            mw, mh = area.width, area.height

        scale = min(self.cnn_width / max(mw, 1), self.cnn_height / max(mh, 1))
        offset_x = max(0, (self.cnn_width - int(mw * scale)) // 2)
        offset_y = max(0, (self.cnn_height - int(mh * scale)) // 2)

        rel_x = pos.x - crop_x
        rel_y = pos.y - crop_y

        x_cnn = (rel_x * scale) + offset_x
        y_cnn = (self.cnn_height - 1) - ((rel_y * scale) + offset_y)

        # Scale CNN coordinates to grid coordinates
        grid_x = int(x_cnn * self.grid_size / self.cnn_width)
        grid_y = int(y_cnn * self.grid_size / self.cnn_height)

        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))

        return grid_y * self.grid_size + grid_x

    def reset(self):
        self.transformed_observation = None
        self.last_game_info = None
        self.last_action = None