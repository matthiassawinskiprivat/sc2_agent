# Implementation Plan - Remove Dilation from Load Action Mask

## Phase 1: Preparation & Testing Strategy [checkpoint: c988819]
- [x] Task: Create a new test file `tests/test_transport_mask_precision.py`.
    - [x] Implement a test that places a single unit and verifies the `get_transport_mask` output area.
    - [x] Assert that the masked area is not expanded by dilation (e.g., check that the number of masked pixels is consistent with a non-dilated point).
- [x] Task: Run the test to confirm it fails with current dilation (Red Phase).
- [x] Task: Conductor - User Manual Verification 'Preparation & Testing Strategy' (Protocol in workflow.md)

## Phase 2: Implementation
- [x] Task: Modify `src/agent/unit_control_agent.py`. 48bd188
    - [x] In `get_transport_mask`, remove the call to `cv2.dilate`.
- [x] Task: Run `tests/test_transport_mask_precision.py` to verify the fix (Green Phase).
- [ ] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)

## Phase 3: Validation & Cleanup
- [ ] Task: Run existing integration tests to ensure no regressions in loading logic.
- [ ] Task: Remove any unused imports or variables if applicable.
- [ ] Task: Conductor - User Manual Verification 'Validation & Cleanup' (Protocol in workflow.md)
