# Implementation Plan - Fix Drop Mechanics Masking

## Phase 1: UnitControlAgent Masking Refinement (TDD) [checkpoint: 43224a8]
- [x] Task: Create failing unit tests in `tests/test_drop_masking_logic.py` for:
    - `get_transport_mask` when the actor is a transport (should see ground units).
    - `get_transport_mask` when the actor is ground (should see transports).
    - `get_unload_mask` with a 3x3 kernel and enemy buildings.
- [x] Task: Update `get_transport_mask` in `src/agent/unit_control_agent.py` to accept an `actor_type` or `is_transport` flag.
- [x] Task: Update `get_unload_mask` in `src/agent/unit_control_agent.py` to use a 3x3 kernel and ensure it handles all passed units correctly.

## Phase 2: SC2Bot Integration [checkpoint: 43224a8]
- [x] Task: Update `SC2Bot.action_masks()` in `src/sc2_bot.py`:
    - Determine if the current active unit is a transport.
    - Pass this context to `get_transport_mask`.
    - Pass both `self.all_units` (or combined list) to `get_unload_mask`.
- [x] Task: Verify with `tests/test_sc2bot_unload_mask.py` and new integration tests.

## Phase 3: Verification & Performance [checkpoint: 5cc8881]

- [x] Task: Run `tests/test_sc2bot_load_mask.py` and ensure it now passes with the new context-aware logic.

- [x] Task: Implement retry logic in `Level0Env` to handle `port=None` launch errors.

- [x] Task: Perform a manual visual check of masks if possible, or use debug logging to verify mask densities.

- [x] Task: Conductor - User Manual Verification (Protocol in workflow.md)
