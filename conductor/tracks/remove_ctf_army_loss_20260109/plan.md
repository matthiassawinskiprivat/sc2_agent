# Plan: Remove Trainee Army Loss Termination Condition in CTF

This plan removes the early termination condition when the trainee (Player 1) loses all combat units in the Capture the Flag environment.

## Phase 1: Test Updates (TDD Red)
- [x] Task: Update `tests/test_ctf_termination_repro.py` to reflect that trainee army loss should NOT terminate. 1ec6c63
- [x] Task: Update `tests/test_ctf_end_conditions_logic.py` to remove p1_army_lost termination expectations. 1ec6c63
- [x] Task: Run updated tests and confirm they fail against current implementation. 1ec6c63
- [~] Task: Conductor - User Manual Verification 'Phase 1: Test Updates' (Protocol in workflow.md)

## Phase 2: Implementation
- [ ] Task: Modify `CaptureTheFlagEnv._check_termination` in `train/envs/capture_the_flag_env.py`.
- [ ] Task: Modify `CaptureTheFlagSingleEnv._async_step` in `train/envs/capture_the_flag_single_env.py`.
- [ ] Task: Modify `CaptureTheFlagSelfPlayEnv._calculate_final_p1_reward` in `train/envs/capture_the_flag_self_play_env.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Verification (TDD Green)
- [ ] Task: Run updated tests `tests/test_ctf_termination_repro.py` and `tests/test_ctf_end_conditions_logic.py` and confirm they pass.
- [ ] Task: Run all CTF integration tests to ensure no regressions.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification' (Protocol in workflow.md)
