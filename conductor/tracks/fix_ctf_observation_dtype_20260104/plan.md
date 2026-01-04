# Plan - Fix CTF Observation Data Type and Scaling

This plan focuses on fixing the "blind agent" bug by correctly scaling and casting observation tensors to `uint8`.

## Phase 1: Logic Implementation

- [x] Task: Update `src/transformers/observation_transformer.py` to scale the final observation array by 255, clip it to [0, 255], and cast it to `np.uint8`. (00d857d)
- [x] Task: Audit `SharedObservationTransformer` and ensure its cache handles the `uint8` data correctly. (00d857d)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Logic Implementation' (Protocol in workflow.md) (00d857d)

## Phase 2: Verification

- [ ] Task: Update `diagnose_ctf_obs.py` to assert that the max value is 255 and the type is `uint8`.
- [ ] Task: Run `diagnose_ctf_obs.py` and confirm the fix.
- [ ] Task: Run existing environment and rendering tests to ensure the type change doesn't break assertions.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification' (Protocol in workflow.md)
