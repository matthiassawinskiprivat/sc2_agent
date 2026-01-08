# Plan: Fix CTF End Conditions and Rewards

## Phase 1: Test Foundations for End Conditions [checkpoint: a29f923]
Establish a robust testing suite to reproduce the premature termination and verify correct end conditions.

- [x] Task: Create reproduction test for premature termination and timeout logic.
- [x] Task: Create unit tests for Pylon destruction (Win/Loss) outcomes.
- [x] Task: Create unit tests for Mutual Destruction (Draw) outcomes.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Test Foundations for End Conditions' (Protocol in workflow.md)

## Phase 2: Refine Termination Logic [checkpoint: 6180048]
Overhaul the logic in `CaptureTheFlagSingleEnv` to strictly follow the specification's end conditions.

- [x] Task: Update `CaptureTheFlagSingleEnv._async_step` to correctly evaluate pylon health and unit counts. (b3ddfc2)
- [x] Task: Ensure `steps_count` is correctly handled and compared against `max_steps`. (b3ddfc2)
- [x] Task: Implement clear logging for termination reasons (Win, Loss, Draw). (b3ddfc2)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Refine Termination Logic' (Protocol in workflow.md)

## Phase 3: Reward Attribution and Configuration [checkpoint: 8428dfb]
Implement the configurable reward/penalty system and ensure correct terminal attribution.

- [x] Task: Update `CaptureTheFlagBot.calculate_reward` (Confirmed: terminal rewards handled in Env for better control)
- [x] Task: Implement `draw_penalty` and `loss_penalty` in the reward logic.
- [x] Task: Verify reward attribution in the final step of Win, Loss, and Draw scenarios.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Reward Attribution and Configuration' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Reward Attribution and Configuration' (Protocol in workflow.md)

## Phase 4: Integration and Stability [checkpoint: 9dd6af3]
Verify the fixes in the full training pipeline.

- [x] Task: Run a 500-step training session to confirm stability and correct episode cycling. (b3ddfc2)
- [x] Task: Validate that `max_steps` correctly truncates long episodes without intermediate resets. (b3ddfc2)
- [x] Task: Conductor - User Manual Verification 'Phase 4: Integration and Stability' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration and Stability' (Protocol in workflow.md)
