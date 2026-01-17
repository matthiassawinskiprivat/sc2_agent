# Implementation Plan - Improve Loading Logic

## Phase 1: Preparation & Testing Strategy
- [x] Task: Create a dedicated test file `tests/test_loading_mechanics.py`.
    - [x] Create a test case for `Medivac` picking up a `Marine`.
    - [x] Create a test case for `Marine` entering a `Medivac`.
    - [x] Create a test case for `Overlord` picking up a `Drone`.
    - [x] Create a test case for `WarpPrism` picking up a `Probe`.
- [x] Task: Run initial tests to confirm current failure/behavior (Expect failures or inconsistent behavior).
- [ ] Task: Conductor - User Manual Verification 'Preparation & Testing Strategy' (Protocol in workflow.md)

## Phase 2: Implementation
- [ ] Task: Refactor `load` action in `src/sc2_bot.py`.
    - [ ] Define the `TRANSPORT_LOAD_ABILITIES` mapping constant.
    - [ ] Implement the `is_transport` check logic.
    - [ ] Implement the logic branch: `Transport` -> `Specific Ability` vs `Unit` -> `Smart`.
    - [ ] Implement the target finding logic (find loadable unit vs. find transport).
- [ ] Task: Verify changes with `tests/test_loading_mechanics.py`.
- [ ] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)

## Phase 3: Validation & Cleanup
- [ ] Task: Run full integration tests to ensure no regressions.
- [ ] Task: Update `tech-stack.md` if any new libraries or patterns were introduced (Unlikely).
- [ ] Task: Conductor - User Manual Verification 'Validation & Cleanup' (Protocol in workflow.md)
