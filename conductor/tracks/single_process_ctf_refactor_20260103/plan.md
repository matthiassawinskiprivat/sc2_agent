# Plan: Single-Process CTF Refactoring

Refactor the Capture the Flag environment to a single-process architecture using a "God-Mode" bot to manage both players.

## Phase 1: Single-Client Environment Foundation [checkpoint: 06caa59]
Scaffold the new environment and migrate the basic setup logic from the multi-client version.

- [x] Task: Create `SingleClientEnv` base class (or equivalent) to manage a single SC2 process (06caa59)
- [x] Task: Write unit tests to verify single process launch and basic connection stability (06caa59)
- [x] Task: Implement `CaptureTheFlagSingleEnv` with basic map loading and initialization (06caa59)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Single-Client Environment Foundation' (Protocol in workflow.md)

## Phase 2: God-Mode Unit Management and Control [checkpoint: 4df8a2b]
Implement the logic to spawn and command units for both players through a single debug connection.

- [x] Task: Write tests for "God-Mode" spawning (verifying units belong to different players) (4df8a2b)
- [x] Task: Implement `_spawn_initial_units` using debug commands for P1 and P2 (4df8a2b)
- [x] Task: Write tests for dual-side commanding (ensuring commands reach units of both owners) (4df8a2b)
- [x] Task: Implement `apply_action` logic to target units based on the current active player turn (4df8a2b)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: God-Mode Unit Management and Control' (Protocol in workflow.md)

## Phase 3: Perspective Flipping and Cycle Logic [checkpoint: 4df8a2b]
Implement the "Unit-Only Flip" for observations and the turn-based cycle logic.

- [x] Task: Write unit tests for `get_observation` with perspective flipping (verifying allied/enemy swap) (4df8a2b)
- [x] Task: Implement perspective-aware observation generation in `CaptureTheFlagSingleEnv` (4df8a2b)
- [x] Task: Implement the turn-based cycle and reward attribution for the active player (4df8a2b)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Perspective Flipping and Cycle Logic' (Protocol in workflow.md)

## Phase 4: Integration and Stress Testing [checkpoint: b3ddfc2]
Verify the full training pipeline and ensure system stability.

- [x] Task: Run integration test for `num_envs` configuration (verify process count) (b3ddfc2)
- [x] Task: Perform a 100-cycle reset stress test to ensure zero process leaks or hangs (b3ddfc2)
- [x] Task: Verify training performance (50k steps) with the new architecture (b3ddfc2)
- [x] Task: Conductor - User Manual Verification 'Phase 4: Integration and Stress Testing' (Protocol in workflow.md)
