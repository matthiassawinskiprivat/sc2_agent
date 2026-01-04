# Specification: Fix CTF Environment Initialization

## 1. Overview
This track addresses a `TypeError` that occurs when the `PipelineManager` attempts to instantiate `CaptureTheFlagEnv`. The issue arises because the `MultiClientEnv` base class constructor does not accept standard project parameters (such as `level_name`, `sc2_process_cls`, etc.) passed by the training pipeline.

## 2. Functional Requirements
- **Constructor Update:** Modify `MultiClientEnv.__init__` to accept `level_name`, `sc2_process_cls`, `client_cls`, `training_conf`, and `**kwargs`.
- **Parameter Storage:** Store `level_name` and `training_conf` as instance attributes for logging and consistency with other environment classes.
- **Dependency Injection:** Properly handle `sc2_process_cls` and `client_cls` to support mock-based testing, following the pattern established in `Level0Env`.
- **Forward Compatibility:** Ensure `**kwargs` are accepted to prevent future `TypeError` if new parameters are added to the pipeline's instantiation call.

## 3. Acceptance Criteria
- `CaptureTheFlagEnv` (and any other class inheriting from `MultiClientEnv`) can be instantiated by `PipelineManager._train_level` without crashing.
- Unit tests confirm that the environment correctly stores the passed parameters and initializes its internal manager with the provided classes.
