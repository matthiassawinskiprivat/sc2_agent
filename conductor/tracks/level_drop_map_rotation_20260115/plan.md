# Implementation Plan - LevelDropEnv Map Discovery and Rotation

Refactor `LevelDropEnv` to support automatic map discovery and per-reset rotation, removing manual configuration and improving training diversity.

## Phase 1: Configuration & Signature Refactor
Clean up the configuration files and the environment constructor to remove the `map_name` parameter.

- [x] Task: Remove `map_name` from configuration cbe9898
    - [x] Update `config/level_drop_0.yaml` to remove the `map_name` key.
- [ ] Task: Refactor `LevelDropEnv` constructor
    - [ ] Remove `map_name` from the `__init__` signature in `train/envs/level_drop_env.py`.
    - [ ] Ensure `super().__init__` is called correctly without the explicit map name.
- [ ] Task: Update existing tests for signature changes
    - [ ] Update `tests/test_level_drop_system.py`.
    - [ ] Update `tests/test_level_drop_config_refactor.py`.
    - [ ] Update `tests/test_additive_config.py` (specifically LevelDrop sections).
- [ ] Task: Conductor - User Manual Verification 'Configuration & Signature Refactor' (Protocol in workflow.md)

## Phase 2: Map Discovery and Rotation Logic
Implement the logic to discover maps in the level-specific directory and cycle through them on every reset.

- [ ] Task: Implement Map Discovery in `LevelDropEnv`
    - [ ] In `__init__`, ensure `self.map_list` is populated via `self._discover_maps()`.
    - [ ] Initialize `self.map_index` using `self.worker_index % len(self.map_list)`.
    - [ ] Set initial `self.map_name` using the calculated `self.map_index`.
- [ ] Task: Override `reset()` for Per-Reset Rotation
    - [ ] Override the `reset` method in `LevelDropEnv`.
    - [ ] Implement logic to increment `self.map_index` (modulo `len(self.map_list)`).
    - [ ] Update `self.map_name` to the next map.
    - [ ] **Critical:** Call `self.close()` to ensure the current SC2 process is terminated, forcing a fresh process launch with the new map.
- [ ] Task: Conductor - User Manual Verification 'Map Discovery and Rotation Logic' (Protocol in workflow.md)

## Phase 3: System Integration & Final Verification
Verify the end-to-end flow and ensure no regressions in training performance or stability.

- [ ] Task: Write Integration Test for Map Rotation
    - [ ] Create `tests/test_level_drop_map_rotation.py`.
    - [ ] Mock the map directory to contain multiple fake `.SC2Map` files.
    - [ ] Instantiate `LevelDropEnv` and call `reset()` multiple times.
    - [ ] Assert that `self.map_name` changes correctly on each reset.
- [ ] Task: Verify Multi-Worker Offset
    - [ ] Add a test case to verify that different `worker_index` values result in different starting maps.
- [ ] Task: Run Full Regression Suite
    - [ ] Run `pytest` for all Level 0, Level 7, and Level Drop tests.
- [ ] Task: Conductor - User Manual Verification 'System Integration & Final Verification' (Protocol in workflow.md)
