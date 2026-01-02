# Plan: Capture the Flag Environment with Self-Play

Based on the specification for `CaptureTheFlagEnv`, this plan outlines the phases for implementing multi-client support, resource-based spawning, and competitive reward logic.

## Phase 1: Infrastructure - Multi-Client Support [checkpoint: 0d3a2d8]
Implement the capability to run two SC2 processes and connect two bots to the same game session.

- [x] Task: Create `MultiClientSC2Manager` utility to handle dual SC2 processes.
- [x] Task: Implement `MultiClientEnv` base class with synchronized `step()` logic.
- [x] Task: Write tests for dual-process launch and synchronization.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure - Multi-Client Support' (Protocol in workflow.md) 0d3a2d8

## Phase 2: Environment - CaptureTheFlagEnv [checkpoint: f2382aa]
Implement the core CTF logic, resource spawning, and observation transformation.

- [x] Task: Create `CaptureTheFlagEnv` inheriting from `MultiClientEnv`.
- [x] Task: Implement resource-based unit spawning logic (Minerals/Vespene).
- [x] Task: Implement Pylon (Flag) spawning with reduced HP and random placement.
- [x] Task: Update `SharedObservationTransformer` to support 7 layers and Pylon highlighting.
- [x] Task: Implement Serial Unit Control cycle for two players.
- [x] Task: Write tests for spawning logic and observation shapes.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Environment - CaptureTheFlagEnv' (Protocol in workflow.md) f2382aa

## Phase 3: Logic - Self-Play and Rewards [checkpoint: b337a5e]
Implement the competitive reward structure and the self-play inference loop.

- [x] Task: Implement `CaptureTheFlagBot` logic for calculating competitive rewards.
- [x] Task: Implement the self-play inference loop (Model predicts for both P1 and P2).
- [x] Task: Configure `config/capture_the_flag.yaml` with CTF parameters.
- [x] Task: Write integration tests for the full CTF game loop and reward attribution.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Logic - Self-Play and Rewards' (Protocol in workflow.md) b337a5e

## Phase 4: Integration and Verification [checkpoint: e1a283b]
Finalize the training pipeline integration and perform end-to-end verification.

- [x] Task: Integrate `CaptureTheFlagEnv` into `train_pipeline.py`.
- [x] Task: Verify that `command_assignment` and `command_objective` layers are correctly highlighted during steps.
- [x] Task: Run a short training session (1000 steps) to ensure stability.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Integration and Verification' (Protocol in workflow.md) e1a283b
