# Specification: Fix Level 3 Training Initialization and Execution Errors

## Overview
This track addresses critical errors encountered when starting the training process for `Level3Env`. These errors included an `AttributeError` due to uninitialized `BotAI` attributes and a `TypeError` caused by incorrectly awaiting synchronous methods.

## Functional Requirements
- **Initialize BotAI Attributes**: Ensure `enemy_race` and other necessary attributes are initialized in `SC2Bot` to support environments bypassing standard `run_game` loops.
- **Correct Method Invocation**: Remove `await` from calls to `self.do()`, as it is a synchronous method in the `burnysc2` library.
- **Multi-Unit Cycle Verification**: Ensure the `Level3Env` correctly cycles through all active units and steps the game simulation only after all actions are submitted.

## Non-Functional Requirements
- **Stability**: Prevent regressions in environment setup and execution.
- **Observability**: Maintain clear logging and comments for educational purposes regarding RL implementation.

## Acceptance Criteria
- [ ] `Level3Env` can be initialized and reset without `AttributeError`.
- [ ] Actions can be applied in `Level3Env` without `TypeError: object bool can't be used in 'await' expression`.
- [ ] Integration test `tests/test_level_3_integration.py` passes, verifying reset, step cycling, and reward calculation.
- [ ] All existing tests pass.

## Out of Scope
- Refactoring the reward logic for `Level3Env` beyond basic verification.
- Implementing the Master Agent or high-level strategic coordination.
