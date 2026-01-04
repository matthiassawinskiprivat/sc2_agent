# Specification - Randomized CTF Pylon Spawning

## Overview
This track addresses the lack of randomization in the multi-client `CaptureTheFlagEnv` (where Pylons are currently hardcoded to specific corners) and ensures that the single-client `CaptureTheFlagSingleEnv` randomization is robust and effective. The goal is to ensure Pylons are spawned at varied, random locations while maintaining a minimum safe distance between them to ensure fair and diverse training scenarios.

## Functional Requirements
*   **Implement Randomization in Multi-Client Env:** Port the randomization logic from `CaptureTheFlagSingleEnv` to `CaptureTheFlagEnv`, replacing the hardcoded `(10, 10)` and `(w-10, h-10)` coordinates.
*   **Robust Distance Check:** Implement a distance validation logic that ensures P1 and P2 Pylons are at least a configurable distance apart. This distance should have a sensible default based on the map's diagonal to prevent Pylons from spawning on top of each other.
*   **Coordinate Fixes:** Audit and fix any grid-alignment or coordinate calculation bugs in the current `get_rand_pos` logic.
*   **Enhanced Logging:** Add detailed logging during the spawning phase to report the chosen Pylon coordinates and the final distance between them.
*   **Preserve Mechanics:** Units will continue to spawn directly on top of their respective Pylons, and Pylon health/shields will remain hardcoded at 10/10.

## Non-Functional Requirements
*   **Consistency:** The spawning and randomization logic must be identical (or as close as possible) between the single-client (God-Mode) and multi-client environments.
*   **Performance:** The randomization loop (attempts to find a valid position) must be optimized to avoid hanging the environment initialization.

## Acceptance Criteria
*   In `CaptureTheFlagEnv`, Pylons no longer spawn at the same fixed coordinates every episode.
*   Pylons are guaranteed to be at least `flag_distance_min` apart.
*   Logs clearly show the coordinates of both Pylons and the calculated distance.
*   Integration tests confirm that Pylons appear in different map quadrants over multiple resets.

## Out of Scope
*   Randomizing unit starting positions (units stay on Pylons).
*   Changing Pylon health or shield values.
*   Modifying the unit pool or resource budget logic.