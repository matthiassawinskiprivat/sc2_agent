# Specification: Agent Performance Optimization Research & Fix

## Overview
The goal of this track is to identify and resolve the primary bottleneck preventing the StarCraft 2 Unit Control Agent from learning efficiently. The primary optimization identified is **Observation Space Pruning**, reducing the 42-layer input to a minimal set required for the current level.

## Functional Requirements
1.  **Deep Audit & Investigation** (Completed):
    *   Identified that the 42-layer observation space is the primary bottleneck.
2.  **Additive Layer Configuration**:
    *   The active observation layers must be definable in the level configuration files (e.g., `config/level_0.yaml`).
    *   The configuration must be **additive**:
        *   Level 0 defines the initial set.
        *   Subsequent levels inherit the previous set and can append new layers.
        *   If a level defines no new layers, the previous set remains active.
3.  **Dynamic Observation Space**:
    *   The `ObservationTransformer` must output a tensor matching ONLY the active layers.
    *   The `Level0Env` (and subclasses) must dynamically update their `observation_space` definition to match the number of active layers.
4.  **Verification**:
    *   Write tests to verify the additive logic (Level 0 -> Level 1).
    *   Verify the observation shape matches the configured layers.

## Non-Functional Requirements
*   **Maintainability**: Ensure code changes remain modular.
*   **Test-Driven Development**: Adhere strictly to the project's TDD workflow.

## Acceptance Criteria
*   [ ] A detailed research report is documented (Done).
*   [ ] Layer configuration is implemented in `config/level_*.yaml`.
*   [ ] `ObservationTransformer` respects the additive configuration.
*   [ ] All new and existing tests pass.
*   [ ] Code coverage for the changed modules is >80%.
