# Plan: Fix Asymmetric Termination Logic

This plan addresses the issue where the CTF environment fails to terminate when the Trainee (Player 1) loses all units, and ensures the "Persistent Objective" remains active for the Opponent (Player 2).

## Phase 1: Diagnosis & Reliable Detection
Improve how the environment detects "army loss" by investigating the gap between mocks and real SC2 unit states.

- [x] Task: TDD - Create a reproduction test that specifically checks unit-count logic against various unit states (e.g., dying units, larvae). 85ac83a
- [x] Task: Research and improve the unit filtering logic in `_check_termination` to be more robust against StarCraft 2's internal unit management. 85ac83a
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Diagnosis & Reliable Detection' (Protocol in workflow.md)

## Phase 2: Termination Logic Fix
Implement the corrected asymmetric termination conditions.

- [x] Task: TDD - Update `_check_termination` tests to verify that P1 loss triggers termination while P2 loss allows continuation. 85ac83a
- [x] Task: Implement the robust unit check and asymmetric branching in `train/envs/capture_the_flag_env.py`. 85ac83a
- [x] Task: Ensure that `loss_penalty` is reliably applied to the final reward when the Trainee army is wiped out. 85ac83a
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Termination Logic Fix' (Protocol in workflow.md)

## Phase 3: Regression & Validation
Ensure the fix doesn't break Pylon-based termination or reward calculation.

- [x] Task: TDD - Verify that Pylon destruction still terminates the game correctly for both sides. 85ac83a
- [x] Task: Integration Test - Run a short session to confirm that the game ends exactly when the Green player disappears from the map. 85ac83a
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Regression & Validation' (Protocol in workflow.md)
