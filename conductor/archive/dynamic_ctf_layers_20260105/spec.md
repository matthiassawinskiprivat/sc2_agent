# Specification: Dynamic Layer Accumulation for CTF Environment

## Overview
Refactor the Capture the Flag (CTF) environments to support dynamic observation layer accumulation using the `ConfigManager`. Currently, CTF environments use a combination of hardcoded "base layers" and manually specified YAML layers. This track will change that behavior so CTF environments automatically inherit all layers defined in the preceding levels of the training pipeline, ensuring the agent retains all previously learned features (such as air unit visibility and DPS heatmaps) without manual configuration in every file.

## Functional Requirements
1.  **Base Class Integration:** Modify `MultiClientEnv` (the base class for CTF) to initialize and manage a `ConfigManager` instance.
2.  **Dynamic Accumulation:** Refactor `CaptureTheFlagEnv` and `CaptureTheFlagSingleEnv` to calculate `active_layers` using `ConfigManager.get_cumulative_active_layers(self.level_name)`.
3.  **Observation Space Parity:** Ensure the `observation_space` and `HierarchicalAgentSystem` are correctly initialized with the total number of accumulated layers.
4.  **YAML Independence:** The system must correctly accumulate layers even if the `active_layers` list in the CTF YAML file is empty or missing (inheriting only from predecessors).
5.  **Documentation:** Update `product.md` to clearly explain how layers are added level-by-level through the curriculum and inherited by the self-play environments.

## Non-Functional Requirements
-   **Stability:** The transition must not break existing training workflows or model weight grafting logic.
-   **Maintainability:** Removing hardcoded layer lists in the environment code to prefer the centralized `pipeline.yaml` order.

## Acceptance Criteria
-   `CaptureTheFlagEnv` successfully initializes with a layer count that matches the cumulative sum of all preceding levels in `pipeline.yaml`.
-   Existing CTF YAML files (which may have empty `active_layers` or specific CTF layers) work correctly without modification.
-   The `product.md` file contains a section documenting the "Dynamic Layer Stack" mechanism.
-   Automated tests verify that layers from `level_0` through `level_10` are present in the CTF observation space.

## Out of Scope
-   Modifying the older `Level0Env` or `Level7Env` (which already use ConfigManager or follow different patterns).
-   Changing the actual transformation logic in `ObservationTransformer`.
