# Plan: Level 7 Allied Damage and Kill Penalties

## Phase 1: Configuration and Bot Logic
- [x] Task: Update `config/level_7.yaml` to include `damage_penalty: -0.5` and `kill_penalty: -10.0`. 7851e93
- [ ] Task: Implement `calculate_allied_attrition_penalty` in `Level7Bot` using a `previous_allied_health` dictionary.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Configuration and Bot Logic' (Protocol in workflow.md)

## Phase 2: Environment Integration
- [ ] Task: Update `Level7Env.step` (or `_async_step`) to fetch and apply the allied attrition penalty from the bot.
- [ ] Task: Ensure the penalty is correctly summed with existing combat rewards and distance rewards.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Environment Integration' (Protocol in workflow.md)

## Phase 3: Verification and Testing
- [ ] Task: Create `tests/test_level_7_allied_penalties.py` to verify penalty calculation logic with mock allied unit damage/death.
- [ ] Task: Run a short debug training session using `debug_train_level_7.py` and confirm negative rewards appear when allied units take damage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification and Testing' (Protocol in workflow.md)
