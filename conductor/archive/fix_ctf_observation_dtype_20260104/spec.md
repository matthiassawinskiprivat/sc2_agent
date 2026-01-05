# Specification - Fix CTF Observation Data Type and Scaling

## Overview
This track addresses a critical bug where the agents are effectively "blinded" because the `ObservationTransformer` returns `float32` values in the `[0, 1]` range, while the environments expect `uint8` values in the `[0, 255]` range. This mismatch causes the observations to be interpreted as nearly all zeros (or ones), leading to the observed behavior where agents target a fixed, arbitrary point on the map's edge.

## Functional Requirements
*   **Centralized Scaling:** Update `ObservationTransformer.transform` to scale all observation layers by `255.0`.
*   **Robust Casting:** Ensure that values are clipped to the `[0, 255]` range before being cast to `np.uint8` to prevent overflow or wrapping.
*   **Unified DType:** All methods in `ObservationTransformer` that return observation data must now return `np.uint8`.
*   **Bot Consistency:** Verify that `CaptureTheFlagBot` and `SC2Bot` no longer need to perform manual casting if they use the transformer.

## Non-Functional Requirements
*   **Performance:** The scaling and casting should be performed using efficient NumPy vectorized operations.
*   **Reliability:** This change must fix the "blind agent" issue, where `obs.max()` was previously only `1.0`.

## Acceptance Criteria
*   The `diagnose_ctf_obs.py` script reports an observation max value of `255` and a `uint8` data type.
*   Agents in the CTF environment no longer exhibit the fixed right-edge targeting behavior.
*   Observation intensity is visually correct when rendered in the debug UI.

## Out of Scope
*   Modifying the neural network architecture.
*   Changing the reward functions.
*   Adjusting the number of observation layers.
