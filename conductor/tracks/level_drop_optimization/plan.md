# Implementation Plan - Level Drop Optimization

## Phase 1: Environment Modification
- [ ] **Step 1:** Modify `train/envs/level_drop_env.py` to import `Point2` from `sc2.position`.
- [ ] **Step 2:** Implement the `_check_stranded_condition` helper method in `LevelDropEnv`.
    -   Filter active units for Transports.
    -   If Transports == 0, find the closest Ground Unit to the objective.
    -   Use `self._client.query_pathing(start, end)` to check for a ground path.
- [ ] **Step 3:** Integrate this check into the main `_async_step` loop.
- [ ] **Step 4:** Add a specific "Stranded Penalty" (optional, but good for reinforcement) or just rely on the existing `loss_penalty`.

## Phase 2: Verification
- [ ] **Step 1:** Create `debug_stranded_check.py` based on `debug_train_level_7.py` but configured for Level Drop 2.
- [ ] **Step 2:** Manually command the bot (or let it run) to lose its transport and verify the episode ends.
