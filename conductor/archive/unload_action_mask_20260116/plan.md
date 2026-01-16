# Implementation Plan - Implement Action Masking for Unload Head

This plan covers the implementation of a spatial action mask for the `unload` head, ensuring safe drops by accounting for terrain margins and building footprints.

## Phase 1: Research & Setup
- [ ] Task: Research `burnysc2` unit properties to identify the most accurate way to represent building footprints (e.g., `radius` vs. grid-based dimensions).
- [ ] Task: Create a base test utility in `tests/masking_test_utils.py` to generate mock SC2 environments with cliffs and buildings for mask validation.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Research & Setup' (Protocol in workflow.md)

## Phase 2: UnitControlAgent Masking Logic (TDD)
- [ ] Task: Write failing unit tests for `UnitControlAgent.get_unload_mask` that verify:
    - Terrain erosion (2-pixel margin from cliffs/voids).
    - Exact exclusion of allied/enemy buildings and resources.
    - Correct scaling and coordinate mapping.
- [ ] Task: Implement `get_unload_mask` in `src/agent/unit_control_agent.py` using `cv2.erode` for the safety margin and `cv2.fillPoly` or similar for exact footprints.
- [ ] Task: Verify implementation with the newly created unit tests.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: UnitControlAgent Masking Logic' (Protocol in workflow.md)

## Phase 3: SC2Bot Integration (TDD)
- [ ] Task: Create failing integration test `tests/test_sc2bot_unload_mask.py` asserting the `unload` head in `SC2Bot.action_masks()` uses the new logic.
- [ ] Task: Update `SC2Bot.action_masks()` in `src/sc2_bot.py` to call `agent.get_unload_mask(units, game_info)` for the `unload` head.
- [ ] Task: Ensure that the `unload` mask is NOT bypassed for flying units (unlike the `move` mask).
- [ ] Task: Verify the integration with tests and check for any performance regressions.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: SC2Bot Integration' (Protocol in workflow.md)
