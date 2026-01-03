# Plan: Fix CTF End Conditions and Rewards

## Phase 1: Test Foundations for End Conditions [checkpoint: a29f923]
Establish a robust testing suite to reproduce the premature termination and verify correct end conditions.

- [x] Task: Create reproduction test for premature termination and timeout logic.
- [x] Task: Create unit tests for Pylon destruction (Win/Loss) outcomes.
- [x] Task: Create unit tests for Mutual Destruction (Draw) outcomes.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Test Foundations for End Conditions' (Protocol in workflow.md)

## Phase 2: Refine Termination Logic
Overhaul the logic in `CaptureTheFlagSingleEnv` to strictly follow the specification's end conditions.

- [ ] Task: Update `CaptureTheFlagSingleEnv._async_step` to correctly evaluate pylon health and unit counts.
- [ ] Task: Ensure `steps_count` is correctly handled and compared against `max_steps`.
- [ ] Task: Implement clear logging for termination reasons (Win, Loss, Draw).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Refine Termination Logic' (Protocol in workflow.md)

## Phase 3: Reward Attribution and Configuration
Implement the configurable reward/penalty system and ensure correct terminal attribution.

- [ ] Task: Update `CaptureTheFlagBot.calculate_reward` to accept terminal state and reward values from config.
- [ ] Task: Implement `draw_penalty` and `loss_penalty` in the reward logic.
- [ ] Task: Verify reward attribution in the final step of Win, Loss, and Draw scenarios.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Reward Attribution and Configuration' (Protocol in workflow.md)

## Phase 4: Integration and Stability
Verify the fixes in the full training pipeline.

- [ ] Task: Run a 500-step training session to confirm stability and correct episode cycling.
- [ ] Task: Validate that `max_steps` correctly truncates long episodes without intermediate resets.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration and Stability' (Protocol in workflow.md)
