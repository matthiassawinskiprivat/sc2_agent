# Specification: Fix train_pipeline TypeError and Improve IPC Error Handling

## Overview
A `TypeError` in `reset_game_state` caused by an unexpected `verify` argument led to environment crashes, which in turn caused `EOFError` and `BrokenPipeError` in the training pipeline. This track fixes the signature mismatch and enhances logging to make such issues easier to diagnose in the future.

## Functional Requirements

### 1. Fix `reset_game_state` Signature
- Update `SC2Bot.reset_game_state` in `src/sc2_bot.py` to accept an optional `verify: bool = True` parameter.
- Ensure all subclasses (`Level3Bot`, `Level7Bot`, etc.) are calling `super().reset_game_state(verify=verify)` or similar correctly.

### 2. Improve IPC Error Handling
- Enhance logging in `train/envs/level_0_env.py` and `train/train_pipeline.py` to better capture and report the root cause when worker processes fail.
- Specifically, ensure that if an `EOFError` occurs during reset, the last known worker log messages or the specific exception that caused the worker to die are prioritized in the output.

### 3. Cleanup
- Verify that `Level3Bot` and `Level7Bot` properly handle empty target lists in their new "greedy" logic (defensive programming).

## Non-Functional Requirements
- **Robustness:** The training pipeline should provide clear feedback when an environment fails to initialize rather than just raising a generic `EOFError`.

## Acceptance Criteria
- [ ] `train_pipeline.py` starts without the `TypeError`.
- [ ] Subprocesses correctly handle the `verify` argument.
- [ ] If a subprocess crashes, the main process provides a more informative error message.
- [ ] Greedily logic is verified to be safe against empty unit/target collections.
