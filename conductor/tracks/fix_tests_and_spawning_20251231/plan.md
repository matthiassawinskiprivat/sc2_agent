# Implementation Plan - Fix Failing Tests and Level 3 Spawning

## Phase 1: Fix Level 3 Spawning and Config Passing
- [x] Task: Verify Level 3 Spawning issue with `tests/test_level_3_spawning.py`
- [x] Task: Update `Level3Env.__init__` to correctly extract `targets` and other env-specific configs from `ConfigManager`
- [x] Task: Modify `Level3Bot` to use the `targets` parameter correctly during `reset_game_state`
- [~] Task: Ensure `PipelineManager` passes the correct `training_conf` to `Level3Env`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Level 3 Spawning' (Protocol in workflow.md)

## Phase 2: Resolve Environment & Logic Test Failures
- [ ] Task: Fix `test_coordinates_manual.py` by updating it to use the new bot/agent access pattern
- [ ] Task: Fix `test_config_reward.py` by aligning expected values with actual configuration in `config/level_2.yaml`
- [ ] Task: Fix `test_env_action_flow.py` by ensuring `_async_step` properly handles the `actions` list and `client.actions` call in the mock environment
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Environment & Logic' (Protocol in workflow.md)

## Phase 3: Resolve Infrastructure & Utilities Test Failures
- [ ] Task: Fix `test_model_utils.py` by updating the assertion for `graft_model_weights` when no change is needed (deep equality vs identity)
- [ ] Task: Fix `test_sync_vec_env_logging.py` by ensuring `SyncVecEnv` methods actually trigger the expected logger calls or update the test to match current behavior
- [ ] Task: Fix `test_watchdog.py` and `test_watchdog_integration.py` by debugging the callback mechanism and ensuring the test environment allows for the timeout to trigger
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Infrastructure & Utilities' (Protocol in workflow.md)

## Phase 4: Resolve Integration Test Failures
- [ ] Task: Fix `test_training_pipeline_integration.py` by updating mocks to provide valid `Race` enum values and matching the unpacking requirements for `env.reset()`
- [ ] Task: Run full test suite to ensure 100% pass rate
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Final Check' (Protocol in workflow.md)
