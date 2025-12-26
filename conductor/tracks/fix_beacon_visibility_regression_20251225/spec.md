# Specification: Fix Beacon Visibility in Observation

## 1. Overview
The agent is failing to learn the task after approximately 240k steps, exhibiting behavior where it runs to a single spot repeatedly. This regression is suspected to be caused by the "Beacon" unit not being correctly rendered or visible in the agent's observations, a problem previously encountered and solved but likely reintroduced during the recent refactor in the `clean-up-with-conductor` branch.

## 2. Problem Description
*   **Symptom:** Agent performance degrades/stalls; it ignores the objective (Beacon).
*   **Hypothesis:** The Beacon unit is missing from the spatial observations. In this project's logic, the Beacon (even if neutral) should be rendered on the `enemy_units` layer to be perceived as a target/objective by the agent.
*   **Suspected Area:** Configuration loading in `train_pipeline.py` and its propagation to `src/transformers/observation_transformer.py`, specifically regarding settings defined in `level_0.yaml` and how the observation layers are populated.

## 3. Functional Requirements
*   **Visibility:** The Beacon unit MUST be rendered on the `enemy_units` spatial layer.
*   **Configuration:** The settings in `level_0.yaml` (and potentially other config levels) MUST be correctly loaded and passed to the `observation_transformer`.
*   **Pipeline:** The `train_pipeline.py` MUST initialize the environment and transformers using the correct, complete configuration.

## 4. Acceptance Criteria
*   **Reproduction:** A script `reproduce_beacon_visibility.py` exists that initializes the environment and asserts whether the Beacon is visible in the `enemy_units` observation layer.
*   **Fix Verification:** The `reproduce_beacon_visibility.py` script confirms the Beacon is visible after the code changes.
*   **Regression Check:** The training pipeline initializes without errors.

## 5. Out of Scope
*   Full training run (240k steps) to prove convergence (we rely on the visibility check).
*   Changes to the model architecture.
