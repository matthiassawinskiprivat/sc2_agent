# Plan: Capture the Flag Environment with Self-Play

Based on the specification for `CaptureTheFlagEnv`, this plan outlines the phases for implementing multi-client support, resource-based spawning, and competitive reward logic.

## Phase 1: Infrastructure - Multi-Client Support [checkpoint: 0d3a2d8]
Implement the capability to run two SC2 processes and connect two bots to the same game session.

- [x] Task: Create `MultiClientSC2Manager` utility to handle dual SC2 processes.
- [x] Task: Implement `MultiClientEnv` base class with synchronized `step()` logic.
- [x] Task: Write tests for dual-process launch and synchronization.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure - Multi-Client Support' (Protocol in workflow.md) 0d3a2d8

## Phase 2: Environment - CaptureTheFlagEnv
Implement the core CTF logic, resource spawning, and observation transformation.

- [ ] Task: Create `CaptureTheFlagEnv` inheriting from `MultiClientEnv`.
- [ ] Task: Implement resource-based unit spawning logic (Minerals/Vespene).
- [ ] Task: Implement Pylon (Flag) spawning with reduced HP and random placement.
- [ ] Task: Update `SharedObservationTransformer` to support 7 layers and Pylon highlighting.
- [ ] Task: Implement Serial Unit Control cycle for two players.
- [ ] Task: Write tests for spawning logic and observation shapes.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Environment - CaptureTheFlagEnv' (Protocol in workflow.md)

## Phase 3: Logic - Self-Play and Rewards
Implement the competitive reward structure and the self-play inference loop.

- [ ] Task: Implement `CaptureTheFlagBot` logic for calculating competitive rewards.
- [ ] Task: Implement the self-play inference loop (Model predicts for both P1 and P2).
- [ ] Task: Configure `config/capture_the_flag.yaml` with CTF parameters.
- [ ] Task: Write integration tests for the full CTF game loop and reward attribution.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Logic - Self-Play and Rewards' (Protocol in workflow.md)

## Phase 4: Integration and Verification
Finalize the training pipeline integration and perform end-to-end verification.

- [ ] Task: Integrate `CaptureTheFlagEnv` into `train_pipeline.py`.
- [ ] Task: Verify that `command_assignment` and `command_objective` layers are correctly highlighted during steps.
- [ ] Task: Run a short training session (1000 steps) to ensure stability.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration and Verification' (Protocol in workflow.md)
