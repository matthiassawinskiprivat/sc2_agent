# Specification - Analyze CTF Right-Edge Movement Bug

## Overview
Agents in the Capture the Flag (CTF) environment (both single and multi-client) are exhibiting a behavior where they immediately move to the right edge of the map. This suggests a systemic issue in coordinate transformation, observation rendering, or reward calculation.

## Goals
1. Identify why agents are choosing the right edge of the map.
2. Verify coordinate mapping symmetry between `ObservationTransformer` (input) and `UnitControlAgent` (output).
3. Verify that the `COMMAND_OBJECTIVE` and `COMMAND_ASSIGNMENT` layers are correctly aligned with the units and their targets.
4. Check for reward bias towards specific coordinates.

## Analysis Steps
*   **Trace Action Conversion:** Log `spatial_idx -> (x_cnn, y_cnn) -> (world_x, world_y)` for acting units.
*   **Visualize Observations:** Use a diagnostic script to dump the `allied_unit_density` and `command_objective` layers to confirm they match the actual game state.
*   **Test Coordinate Invariance:** Run a test with a unit at a known position and a target at a known position, then verify that the agent's preferred action (greedy) points in the correct direction.

## Acceptance Criteria
*   Root cause identified and verified with a reproduction script.
*   Corrective action implemented.
