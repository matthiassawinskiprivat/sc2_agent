# Specification - Fix CTF Pylon Rendering

## Overview
This track fixes several rendering issues in the 'Capture the Flag' (CTF) environments. Pylons are currently invisible on the `command_objective` layer in the Multi-Client environment when in the fog of war. Furthermore, Pylons (being structures) are currently excluded from standard unit-based layers like density and health, making them invisible to the agent except through specific objective logic.

## Functional Requirements
*   **Full Visibility:** Enable `debug_show_map()` for both clients in `CaptureTheFlagEnv` to ensure the entire map, including enemy Pylons, is always visible to the bots.
*   **Structure Rendering:** Update `ObservationTransformer.transform` to include `allied_structures` and `enemy_structures` in the unit rendering loop. This ensures Pylons appear on density, biological/mechanical/armored, health, and other unit-specific layers.
*   **Persistent Objective:** Maintain the behavior in `CaptureTheFlagBot` where the `command_objective` layer always highlights the position of the enemy Pylon.
*   **Accurate Pixel Mapping:** Ensure that Pylon radii are correctly scaled when rendered on the observation layers.

## Non-Functional Requirements
*   **Consistency:** Pylon rendering must be identical between single-client and multi-client environments.
*   **Performance:** Including structures in the rendering loop should not significantly impact the observation transformation speed.

## Acceptance Criteria
*   In Multi-Client CTF, the enemy Pylon is visible on the `command_objective` layer from the start of the episode.
*   Pylons are visible on the `allied_unit_density` and `enemy_unit_density` layers.
*   Pylon health is correctly reflected on the health layers.
*   The `command_objective` layer correctly highlights the enemy Pylon position in both environments.

## Out of Scope
*   Adding new observation layers.
*   Modifying Pylon health or shield values.
