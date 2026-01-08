# Specification: Fix Space Mismatches in Capture the Flag Environment

## Overview
This track addresses a regression where the `CaptureTheFlagEnv` (Level 11) fails to maintain compatibility with the model trained in Level 7. Specifically, it violates the "Strictly Additive" rule for observation layers and is missing the secondary action grid (Attack command) introduced in Level 7.

## Functional Requirements
1.  **Strictly Additive Observation Layers**:
    *   The environment must ensure that the observation space contains all layers defined in previous levels (0-10).
    *   Any new layers defined for Capture the Flag must be appended to the existing stack, never replacing or removing old ones.
    *   The mismatch (9 layers in Level 7 vs 7 layers in CTF) must be resolved by ensuring CTF provides the full 9+ layer stack.

2.  **Action Space Restoration**:
    *   The action space in `CaptureTheFlagEnv` must be restored to 32768 (two 128x128 grids).
    *   The first 16384 actions should represent the "Move" command.
    *   The second 16384 actions should represent the "Attack" command, maintaining parity with the Level 7 action head.

3.  **Bot Logic Update**:
    *   `CaptureTheFlagBot` must be updated to correctly interpret actions from both grids (Move vs Attack).

## Technical Constraints
*   **Weight Compatibility**: The fixes must allow `MaskablePPO` to load the Level 10/Level 7 weights into the CTF environment without triggering a "Space Mismatch" warning or requiring invasive weight grafting for these specific dimensions.

## Acceptance Criteria
*   The `train_pipeline` starts the `capture_the_flag` level without "Space mismatch detected" logs for Observation or Action spaces.
*   Observation shape is at least `(9, 128, 128)`.
*   Action space size is exactly `32768`.
*   The agent can successfully issue both Move and Attack commands in the CTF environment.

## Out of Scope
*   Adding new gameplay mechanics to CTF beyond fixing the interface mismatches.
*   Retraining previous levels.
