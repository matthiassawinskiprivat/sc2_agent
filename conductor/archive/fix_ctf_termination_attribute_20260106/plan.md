# Plan: Fix AttributeError and Termination Logic Final

This plan restores library compatibility for the `CaptureTheFlagBot` and ensures that army-loss termination is robust and reliable by preventing property shadowing and refining the unit filtering.

## Phase 1: Restore BotAI Compatibility
Restore essential internal variables required by the StarCraft II library while ensuring dynamic properties remain unshadowed.

- [x] Task: TDD - Create a test verifying that `CaptureTheFlagBot` initializes with required library attributes (e.g., `distance_calculation_method`) and that `units` is not an instance attribute. df8d017
- [x] Task: Update `CaptureTheFlagBot.__init__` to call `_initialize_variables()` and then `del` shadowing instance attributes. df8d017
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Restore BotAI Compatibility' (Protocol in workflow.md)

## Phase 2: Robust Army Detection & Logic
Refine the army-loss detection to use `type_id` and correctly handle non-combat units and asymmetric roles.

- [x] Task: TDD - Update unit detection tests to use `type_id` and verify that Workers and Overlords are correctly ignored. df8d017
- [x] Task: Implement enhanced `_has_combat_units` using `type_id` filtering in `CaptureTheFlagEnv`. df8d017
- [x] Task: Verify role-agnostic termination branching in `_check_termination` (Trainee loss = Done). df8d017
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Robust Army Detection & Logic' (Protocol in workflow.md)

## Phase 3: Environment Sanity & Integration
Add runtime checks to prevent regressions and verify the full training pipeline startup.

- [x] Task: Add sanity checks to `reset()` in `CaptureTheFlagEnv` and `CaptureTheFlagSingleEnv` to log property types. df8d017
- [x] Task: Integration Test - Verify `train_pipeline` startup without `AttributeError`. df8d017
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Environment Sanity & Integration' (Protocol in workflow.md)