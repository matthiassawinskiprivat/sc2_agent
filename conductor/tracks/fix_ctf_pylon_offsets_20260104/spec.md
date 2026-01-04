# Specification - Fix CTF Pylon Coordinate Offsets

## Overview
This track fixes a coordinate system mismatch in the 'Capture the Flag' (CTF) environments. Currently, Pylon positions are randomized based on the dimensions of the `playable_area`, but these relative coordinates are applied as absolute map coordinates. This causes Pylons to always cluster in the bottom-left quadrant of the map, regardless of the map's actual size or offset. The fix involves adding the `playable_area` origin offsets (x, y) to the randomized positions.

## Functional Requirements
*   **Coordinate Offset Correction:** Update `CaptureTheFlagEnv` and `CaptureTheFlagSingleEnv` to correctly apply the `playable_area.x` and `playable_area.y` offsets to all randomized Pylon and unit coordinates.
*   **Playable Area Adherence:** Ensure that all spawned Pylons and units are placed strictly within the absolute map boundaries defined by the `playable_area`.
*   **Distance Constraint:** Maintain strict adherence to the `flag_distance_min` configuration, calculated using the final absolute coordinates.
*   **Simplified Margin:** Use a minimal margin (e.g., 1.0 or 2.0) to ensure Pylons aren't clipped by map edges, while prioritizing the use of the entire playable field as requested.

## Non-Functional Requirements
*   **Parity:** Both single-client (God-Mode) and multi-client environments must use the same coordinate transformation logic.
*   **Robustness:** The logic should handle maps with non-zero origins (standard in many SC2 maps).

## Acceptance Criteria
*   In both environments, Pylons can now spawn in any part of the map, including the Top-Right corner.
*   The logs correctly report absolute map coordinates for Pylon positions.
*   Pylons never spawn outside the `playable_area`.
*   The distance between Pylons is verified to be >= `flag_distance_min`.

## Out of Scope
*   Adding new randomization algorithms (e.g., quadrant-based).
*   Changes to pathing or movement logic.
