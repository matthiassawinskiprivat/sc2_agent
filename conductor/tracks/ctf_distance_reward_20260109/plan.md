# Plan: CTF Distance Reward Integration

This plan implements a distance-based reward signal for units in the Capture the Flag environment, encouraging them to approach the enemy pylon.

## Phase 1: Test Foundations (TDD)
- [x] Task: Create unit tests in `tests/test_ctf_distance_reward.py` to validate reward logic: 1b367ea
    - Test forward progress (positive reward).
    - Test retreating (negative reward).
    - Test multi-unit isolation (Unit A's movement shouldn't affect Unit B's baseline).
    - Test configuration scaling via `distance_multiplier`.
- [x] Task: Run the new tests and confirm they fail (Red Phase). 1b367ea
- [~] Task: Conductor - User Manual Verification 'Phase 1: Test Foundations' (Protocol in workflow.md)

## Phase 2: Core Implementation
- [ ] Task: Update `CaptureTheFlagBot.__init__` in `src/agent/ctf_bot.py` to initialize a `prev_dist_per_unit` dictionary for tracking.
- [ ] Task: Modify `CaptureTheFlagBot.calculate_reward` to:
    - Retrieve the active unit's distance to the enemy pylon.
    - Calculate reward based on `distance_multiplier` and distance delta.
    - Persist the new distance for that specific unit tag.
- [ ] Task: Update `config/capture_the_flag_0.yaml` (and other CTF configs if applicable) to include `distance_multiplier: 0.1` in the `rewards` section.
- [ ] Task: Verify that all tests in `tests/test_ctf_distance_reward.py` now pass (Green Phase).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Quality Assurance
- [ ] Task: Run existing CTF integration tests (e.g., `tests/test_capture_the_flag_env.py`) to ensure no regressions.
- [ ] Task: Perform a final code review to ensure comments and documentation (especially for RL learning purposes) are clear.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Quality Assurance' (Protocol in workflow.md)
