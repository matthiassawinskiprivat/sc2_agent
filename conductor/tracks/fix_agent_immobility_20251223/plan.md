# Plan - Fix Agent Immobility and Action Transmission

## Phase 1: Environment Fixes (Action Transmission)
- [x] Task 1: Create a failing unit test in `tests/test_env_action_flow.py` that verifies `Level0Env` fails to send actions to the client.
- [x] Task 2: Implement explicit `self._client.actions()` call and action clearing in `Level0Env`.
- [x] Task 3: Verify the fix with the unit test created in Task 1. (Unit test passes, integration tests now pass with dependency injection).
- [x] Task 4: Audit `train/envs/` for any other environment classes (Level 1, Level 2) and apply similar fixes if they don't inherit from `MinimalSC2Env`. (Only `MinimalSC2Env` found, which is a test harness).
- [ ] Task: Conductor - User Manual Verification 'Environment Fixes' (Protocol in workflow.md)

## Phase 2: Action Masking Refinement
- [ ] Task 1: Create a unit test `tests/test_action_mask_logic.py` to verify the current behavior of `UnitControlAgent.get_action_mask` against known valid/invalid coordinates.
- [ ] Task 2: Refine the masking logic in `src/agent/unit_control_agent.py` to ensure it correctly identifies reachable/valid movement targets.
- [ ] Task 3: Update and pass tests for the refined masking logic.
- [ ] Task: Conductor - User Manual Verification 'Action Masking Refinement' (Protocol in workflow.md)

## Phase 3: Integration & System Verification
- [ ] Task 1: Create an integration script or use `debug_train.py` to verify the agent moves in a live StarCraft II instance (no model, random actions).
- [ ] Task 2: Run the full test suite (`pytest`) to ensure no regressions were introduced.
- [ ] Task: Conductor - User Manual Verification 'Integration & System Verification' (Protocol in workflow.md)
