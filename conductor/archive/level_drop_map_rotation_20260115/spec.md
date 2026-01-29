# Specification - LevelDropEnv Map Discovery and Rotation

## Overview
Refactor the `LevelDropEnv` to eliminate manual map configuration and implement an automated, curriculum-aware map discovery and rotation system. This ensures that training on "Level Drop" (and its variations like `level_drop_0`, `level_drop_1`, etc.) automatically utilizes all relevant maps, improving generalization through diverse scenario exposure.

## Functional Requirements
1.  **Automated Map Discovery:**
    *   On initialization, `LevelDropEnv` must scan the directory `train/maps/<level_name>/` (where `<level_name>` corresponds to the training configuration name, e.g., `level_drop_0`).
    *   Only `.SC2Map` files will be considered.
    *   Discovered maps must be sorted alphabetically to ensure deterministic indexing across multiple parallel workers.
2.  **Parameter Refactoring:**
    *   Remove the `map_name` parameter from the `LevelDropEnv.__init__` method signature.
    *   Remove the `map_name` key from `level_drop_0.yaml` (and any future `level_drop_X.yaml` files).
3.  **Multi-Worker Diversity:**
    *   Upon initialization, each environment worker must set its starting `map_index` to `worker_index % num_maps`. This ensures that parallel environments begin on different maps.
4.  **Per-Reset Map Rotation:**
    *   Every call to `reset()` must advance the `map_index` to the next map in the discovered list (cycling back to the first map after the last one).
    *   `self.map_name` must be updated to reflect the new selection.
    *   To ensure the map change takes effect in StarCraft II, the environment must trigger a "hard reset" (closing the current process/connection) whenever the map changes.

## Non-Functional Requirements
*   **Inheritance Integrity:** Leverage and respect existing logic in `Level0Env` for process management and map resolution where possible.
*   **Logging:** Provide clear log entries when a map change is triggered (e.g., `[LevelDropEnv] Rotating to map: DropOperations_v2`).

## Acceptance Criteria
1.  `LevelDropEnv` initializes without a `map_name` parameter.
2.  Maps are successfully discovered from the correct sub-directory in `train/maps/`.
3.  In a multi-worker setup (e.g., 4 workers, 2 maps), workers 0 and 2 start on map A, while workers 1 and 3 start on map B.
4.  After an episode ends and `reset()` is called, the next map in the sequence is loaded.
5.  All existing tests for `LevelDropEnv` pass or are updated to reflect the new signature.

## Out of Scope
*   Randomizing map order (sorting is required for deterministic behavior).
*   Adding map discovery to other environments like `Level7Env` (this refactor is specific to the `LevelDrop` curriculum for now).
