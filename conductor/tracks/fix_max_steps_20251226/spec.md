# Specification: Fix `max_steps` configuration in training environment

## Overview
The `train_pipeline` uses YAML files in `/config` to configure the training level environment. The configuration contains a `max_steps` parameter intended to define when the environment should be soft reset (terminating/truncating the episode). Currently, this parameter is ignored, and the environment defaults to 300 steps.

## Functional Requirements
- **Configuration Propagation:** The `max_steps` value defined in `config/level_*.yaml` must be correctly extracted by `PipelineManager` in `train/train_pipeline.py`.
- **Environment Initialization:** The `max_steps` value must be passed to the `Level0Env` constructor during environment creation in `train/train_pipeline.py`.
- **Environment Logic:** `Level0Env` must use the passed `max_steps` value to determine when an episode should be truncated in its `step()` method.

## Non-Functional Requirements
- **Robustness:** If `max_steps` is missing from the configuration, it should fall back to a sensible default (e.g., 300).
- **Maintainability:** Ensure that environment parameters are passed clearly and consistently.

## Acceptance Criteria
- [ ] `Level0Env` correctly receives the `max_steps` value from the YAML configuration.
- [ ] The `step()` method in `Level0Env` uses this dynamic `max_steps` value instead of the hardcoded default.
- [ ] A unit test verifies that `Level0Env` initialized with a specific `max_steps` value correctly truncates after that many steps.

## Out of Scope
- Modifying other environment parameters unless they are directly related to the `max_steps` issue.
- Significant refactoring of the training pipeline beyond what is necessary to fix this bug.
