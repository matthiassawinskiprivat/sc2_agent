# Specification - Fix Capture the Flag Unit Spawning

## Overview
This track addresses a bug where the 'Capture the Flag' (CTF) environment fails to spawn the full variety of units specified in the `unit_pool` configuration. The root cause is a mismatch between configuration keys in YAML files and the internal parameter names in the environment classes, as well as incomplete hardcoded unit costs in the multi-client environment.

## Functional Requirements
*   **Standardized Configuration:** Update `config/capture_the_flag.yaml` and other relevant configs to use `mineral_budget` and `vespene_budget` instead of `minerals` and `vespene`, matching internal class attributes.
*   **Robust Config Parsing:** Update `CaptureTheFlagEnv` and `CaptureTheFlagSingleEnv` to correctly extract parameters from the flattened `training_conf` dictionary provided by the `PipelineManager`.
*   **Unified Unit Data:** Update `CaptureTheFlagEnv` to load unit costs from `data/unit_data.json` instead of using a limited, hardcoded `UNIT_COSTS` dictionary. This ensures consistency with `CaptureTheFlagSingleEnv` and supports all game units.
*   **Spawning Logic Fix:** Ensure `_calculate_spawns` in both environments correctly identifies affordable units based on the provided budget and pool.

## Non-Functional Requirements
*   **Consistency:** Spawning behavior must be identical between single-client (God-Mode) and multi-client CTF environments.
*   **Maintainability:** Centralizing unit data in `unit_data.json` reduces the risk of future discrepancies.

## Acceptance Criteria
*   The `unit_pool` specified in the configuration (e.g., Zergling, Roach, Marine, Reaper, Zealot) is correctly respected.
*   Units requiring Vespene (like Roaches and Reapers) are spawned when a sufficient vespene budget is provided.
*   Regression tests confirm that `mineral_budget`, `vespene_budget`, and `unit_pool` are correctly loaded from the training configuration.
*   No `KeyError` occurs when using any valid SC2 combat unit in the `unit_pool`.

## Out of Scope
*   Adding new unit types to the game extractor.
*   Modifying the RL model architecture.
*   Changes to the actual combat or movement logic.
