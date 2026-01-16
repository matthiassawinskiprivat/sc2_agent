# Specification - Implement Action Masking for Unload Head

## Overview
Currently, the `unload` action head in `SC2Bot` is unmasked, which can lead to agents attempting to unload units on unpathable terrain, cliffs, or directly on top of buildings. This track implements a spatial action mask for the `unload` head that ensures units are dropped only on valid, pathable ground with a safety margin from cliff edges, while respecting the exact footprints of buildings and resources.

## Functional Requirements

### 1. Spatial Masking for Unload
Implement a method `get_unload_mask` in `UnitControlAgent` (and expose it via `SC2Bot`) that generates a boolean mask for the `unload` action head.

### 2. Terrain Safety Margin
- The mask must exclude unpathable terrain (cliffs, voids).
- A **2-pixel safety margin** must be applied to the pathable area. This means the allowed drop zone should be "eroded" so that units are not dropped too close to the edge of walkable ground.

### 3. Exact Footprints for Entities
The following entities must be excluded from the mask using their **exact footprint** (no safety margin inflation):
- **Allied Buildings**
- **Visible Enemy Buildings**
- **Neutral Resource Patches** (Minerals and Vespene Geysers)
- **Neutral Destructible Rocks/Debris**

### 4. Implementation Logic
To achieve the "safety margin for terrain but exact for buildings" requirement:
1. Start with the current pathing grid.
2. Temporarily "fill" the footprints of all buildings, resources, and destructibles to make them appear pathable.
3. Apply erosion (2 pixels) to this "terrain-only" representation.
4. Finally, mask out (set to False) the exact footprints of all buildings, resources, and destructibles.

### 5. Integration
- Update `SC2Bot.action_masks()` to call the new `get_unload_mask` for the `unload` head.
- Ensure the mask is correctly scaled and resized to match the `grid_size` used by the agent.

## Non-Functional Requirements
- **Performance:** Use efficient NumPy and OpenCV operations (like `cv2.erode` and `cv2.fillPoly`) to minimize impact on the game loop.
- **Accuracy:** Ensure coordinate conversion between SC2 world coordinates and the CNN/Action grid is precise.

## Acceptance Criteria
- [ ] A new test `tests/test_sc2bot_unload_mask.py` verifies that the `unload` mask is applied.
- [ ] Visual verification (via logs or debug UI if possible) shows the unload mask eroded away from cliff edges but strictly following building boundaries.
- [ ] The agent no longer attempts to unload units on invalid terrain or inside buildings.
