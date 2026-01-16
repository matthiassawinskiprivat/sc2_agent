# Implementation Plan - Implement Action Masking for Load Head

## Phase 1: Core Logic in UnitControlAgent [checkpoint: aca9d57]
- [x] Task: Implement `get_transport_mask` in `UnitControlAgent`. [aca9d57]
    - [x] Create failing test `tests/test_transport_mask.py` that asserts a mask is generated near mocked transports.
    - [x] Add `get_transport_mask` to `UnitControlAgent` which identifies transport positions and applies dilation.
    - [x] Verify test passes.
- [ ] Task: Conductor - User Manual Verification 'Core Logic' (Protocol in workflow.md)

## Phase 2: Integration into SC2Bot
- [ ] Task: Integrate transport mask into `SC2Bot.action_masks`.
    - [ ] Create failing test `tests/test_sc2bot_load_mask.py` asserting the `load` head uses the new mask.
    - [ ] Update `SC2Bot.action_masks()` to call `get_transport_mask` for the `load` head.
    - [ ] Verify test passes.
- [ ] Task: Conductor - User Manual Verification 'Integration' (Protocol in workflow.md)

## Phase 3: Final Integration & Quality Gate
- [ ] Task: Full verification and regression check.
    - [ ] Run all existing tests.
    - [ ] Verify no regressions in `move` masking.
- [ ] Task: Final Quality Gate.
    - [ ] Check code coverage for new masking logic.
    - [ ] Ensure adherence to project code style.
- [ ] Task: Conductor - User Manual Verification 'Final Verification' (Protocol in workflow.md)
