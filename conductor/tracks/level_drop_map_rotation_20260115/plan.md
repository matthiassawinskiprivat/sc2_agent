# Implementation Plan - LevelDropEnv Map Discovery and Rotation

Refactor `LevelDropEnv` to support automatic map discovery and per-reset rotation, removing manual configuration and improving training diversity.

## Phase 1: Configuration & Signature Refactor
Clean up the configuration files and the environment constructor to remove the `map_name` parameter.

- [x] Task: Remove `map_name` from configuration cbe9898
    - [x] Update `config/level_drop_0.yaml` to remove the `map_name` key.
- [x] Task: Refactor `LevelDropEnv` constructor dd16781
    - [x] Remove `map_name` from the `__init__` signature in `train/envs/level_drop_env.py`.
    - [x] Ensure `super().__init__` is called correctly without the explicit map name.
- [x] Task: Update existing tests for signature changes e822b2a
    - [x] Update `tests/test_level_drop_system.py`.
    - [x] Update `tests/test_level_drop_config_refactor.py`.
    - [x] Update `tests/test_additive_config.py` (specifically LevelDrop sections).
- [x] Task: Conductor - User Manual Verification 'Configuration & Signature Refactor' (Protocol in workflow.md) [checkpoint: 241a544]

## Phase 2: Map Discovery and Rotation Logic
Implement the logic to discover maps in the level-specific directory and cycle through them on every reset.

- [x] Task: Implement Map Discovery in `LevelDropEnv` 5424169
    - [x] In `__init__`, ensure `self.map_list` is populated via `self._discover_maps()`.
    - [x] Initialize `self.map_index` using `self.worker_index % len(self.map_list)`.
    - [x] Set initial `self.map_name` using the calculated `self.map_index`.
- [x] Task: Override `reset()` for Per-Reset Rotation 5424169
    - [x] Override the `reset` method in `LevelDropEnv`.
    - [x] Implement logic to increment `self.map_index` (modulo `len(self.map_list)`).
    - [x] Update `self.map_name` to the next map.
    - [x] **Critical:** Call `self.close()` to ensure the current SC2 process is terminated, forcing a fresh process launch with the new map.
- [x] Task: Conductor - User Manual Verification 'Map Discovery and Rotation Logic' (Protocol in workflow.md) [checkpoint: 241a544]

## Phase 3: System Integration & Final Verification
Verify the end-to-end flow and ensure no regressions in training performance or stability.

- [x] Task: Write Integration Test for Map Rotation 3bd6530
    - [x] Create `tests/test_level_drop_map_rotation.py`.
    - [x] Mock the map directory to contain multiple fake `.SC2Map` files.
    - [x] Instantiate `LevelDropEnv` and call `reset()` multiple times.
    - [x] Assert that `self.map_name` changes correctly on each reset.
- [x] Task: Verify Multi-Worker Offset 3bd6530
    - [x] Add a test case to verify that different `worker_index` values result in different starting maps.
- [x] Task: Run Full Regression Suite 3bd6530
- [x] Task: Conductor - User Manual Verification 'System Integration & Final Verification' (Protocol in workflow.md) [checkpoint: 241a544]
- [ ] Task: Conductor - User Manual Verification 'System Integration & Final Verification' (Protocol in workflow.md)
