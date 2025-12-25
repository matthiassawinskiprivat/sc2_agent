# Specification: Fix AttributeError: raw_data in Training Pipeline

## Overview
The training pipeline crashes with `AttributeError: raw_data` during environment reset in `train/envs/level_0_env.py`. This error occurs specifically when initializing `sc2.game_state.GameState` with the observation received from the SC2 client. This track aims to identify the cause of this data mismatch and correct the environment code to ensure stable game initialization.

## Functional Requirements
1.  **Investigate Observation Data:** Determine the actual structure of the object returned by `self._client.observation()` to understand why `raw_data` is missing or inaccessible.
2.  **Fix GameState Initialization:** Modify `level_0_env.py` (and any other affected files like `src/sc2_bot.py`) to correctly instantiate `GameState`. This might involve handling the response object differently or wrapping it correctly.
3.  **Ensure Library Compatibility:** Verify that the fix aligns with the conventions of the currently installed `burnysc2` library version.

## Non-Functional Requirements
1.  **Stability:** The fix must reliably prevent the crash across multiple environment resets.
2.  **Performance:** The fix should not introduce significant latency during the reset phase.

## Acceptance Criteria
1.  The `train_pipeline` command executes without triggering `AttributeError: raw_data`.
2.  The SC2 game client launches, and the bot successfully completes the reset phase and enters the game loop.
3.  The `test_action_mask_validity.py` (and other relevant tests) pass successfully.

## Out of Scope
1.  Major refactoring of the action mask logic itself (unless it is the direct cause of the observation corruption).
2.  Upgrading the `burnysc2` library, unless the current version is proven to be broken/incompatible.
