# Specification: Fix Premature Episode Termination

## Overview
After the "Individual Unit Reward Refactoring," the agent is terminating episodes and resetting the environment after only a single step (or very few steps). This is caused by logic errors in the environment's `_async_step` function, where the `done` flag is incorrectly triggered when units are removed from the map or when the unit cycle finishes. This prevents the RL algorithm from collecting sufficient transition data for training.

## Functional Requirements
- **Robust Unit Cycling:** Refactor `Level3Env._async_step` and `Level7Env._async_step` to correctly manage `current_unit_index` even if units die or are removed mid-cycle.
- **Correct Termination Logic:** 
    - Ensure `done=True` is only returned if:
        - All targets (beacons) are collected (Success).
        - All allied units are destroyed (Failure).
    - If the current unit cycle finishes but units/targets remain, the environment must correctly prepare the next observation for the first unit of the next cycle and return `done=False`.
- **State Consistency:** Ensure `self.active_unit_tags` is updated only at the start of a cycle to maintain a stable iteration path, or handled defensively during the step.

## Acceptance Criteria
- [ ] Agent episodes last for multiple steps, allowing for proper policy updates.
- [ ] The `done` flag is never `True` in the middle of a unit cycle unless the mission is over.
- [ ] Worker logs show episodes progressing naturally rather than resetting every few seconds.
- [ ] Verified via tests that unit removal mid-cycle correctly advances to the next unit or starts a new cycle without ending the episode.

## Out of Scope
- Fixing the `WSMessageTypeError` directly (assuming it is a side effect of rapid resets).
- Modifying reward calculation math.
