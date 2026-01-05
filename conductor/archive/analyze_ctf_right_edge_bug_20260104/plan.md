# Plan - Analyze CTF Right-Edge Movement Bug

## Phase 1: Diagnostic Instrumentation

- [x] Task: Instrument `UnitControlAgent.convert_spatial_action_to_world` with detailed logging of the transformation steps.
- [x] Task: Instrument `CaptureTheFlagBot.get_observation` to log the coordinates of units and targets before they are passed to the transformer.
- [x] Task: Create a visualization script that saves observation layers as images for manual inspection.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Diagnostic Instrumentation' (Protocol in workflow.md)

## Phase 2: Analysis and Fix

- [x] Task: Analyze the logged data from Phase 1 to identify discrepancies between game world coordinates and observation pixels.
- [x] Task: Verify that `playable_area` offsets are handled identically in both directions (World->Pixel and Pixel->World).
- [x] Task: Implement the fix for the identified discrepancy.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Analysis and Fix' (Protocol in workflow.md)

## Phase 3: Verification

- [x] Task: Run CTF environments and confirm agents no longer target the right edge. (20bdfca)
- [x] Task: Run all coordinate-related unit tests. (20bdfca)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Verification' (Protocol in workflow.md) (20bdfca)
