# Specification: Fix Beacon Rendering in Observation

## 1. Overview
The agent is currently failing to learn simple navigation tasks, exhibiting random movement that converges to a single map point. This behavior strongly suggests the agent is "blind" to the target objective (the Beacon). The suspected root cause is that the Beacon unit (confirmed as `UnitTypeId.BEACON_TERRANSMALL`) is not being correctly rendered onto the `ENEMY_UNIT_DENSITY` observation layer during training.

## 2. Problem Diagnosis
*   **Symptom:** Agent fails to converge on the target, effectively guessing a static optimal location.
*   **Suspected Cause:** The `observation_transformer.py` may have logic that filters out the Beacon or fails to render it if specific flags (e.g., `render_beacon`) are not correctly propagated from the training configuration.
*   **Critical Constraint:** The debugging process must ensure that the observation pipeline is initialized *exactly* as it is during the actual training loop (`train_pipeline.py`), to avoid "works on my machine/debug script" false positives.

## 3. Functional Requirements
1.  **Verification:**
    *   Create a reproduction test case that initializes the environment and agent exactly as the training pipeline does.
    *   Assert that the `ENEMY_UNIT_DENSITY` layer in the returned observation contains non-zero values corresponding to the Beacon's position.
2.  **Fix:**
    *   Ensure the `render_beacon` configuration (or equivalent) is correctly passed from the Environment/Config down to the `ObservationTransformer`.
    *   Modify `src/transformers/observation_transformer.py` if necessary to ensure the Beacon is reliably rendered when intended.
3.  **Validation:**
    *   The reproduction test case must pass (Beacon visible).
    *   (Optional but recommended) A short training run shows the agent's reward improving or movement distribution changing, confirming it now "sees" the target.

## 4. Technical Details
*   **Target Unit:** `UnitTypeId.BEACON_TERRANSMALL`
*   **Target Layer:** `ENEMY_UNIT_DENSITY`
*   **Relevant Files:**
    *   `src/transformers/observation_transformer.py`: Contains the rendering logic.
    *   `train/envs/level_0_env.py`: Initializes the bot and potentially passes config.
    *   `src/sc2_bot.py`: Manages the actual SC2 unit objects.

## 5. Acceptance Criteria
*   [ ] A regression test exists that fails when the beacon is invisible and passes when it is visible.
*   [ ] The `ENEMY_UNIT_DENSITY` layer is confirmed to show the Beacon in the standard training configuration.
