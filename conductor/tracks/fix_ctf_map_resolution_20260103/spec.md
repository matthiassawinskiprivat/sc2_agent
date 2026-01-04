# Specification: Fix CTF Map Resolution and Attributes

## 1. Overview
This track fixes two issues in `MultiClientEnv`:
1.  **Map Resolution:** The environment was failing to find project-specific maps because it was using standard library resolution only.
2.  **AttributeError:** `observation_space` and `action_space` were incorrectly defined inside a helper method instead of the constructor, causing `Monitor` to crash.

## 2. Functional Requirements
- **Map Resolution:** Implement `_discover_maps` and `_resolve_map` in `MultiClientEnv` to support maps in `train/maps/<level_name>/`.
- **Constructor Fix:** Ensure `observation_space` and `action_space` are initialized in `__init__`.
- **Logging Integration:** Ensure all setup steps are logged for easier debugging.

## 3. Acceptance Criteria
- `CaptureTheFlagEnv` can be wrapped by `Monitor` without `AttributeError`.
- Maps in `train/maps/capture_the_flag/` are correctly discovered and resolved.
- Automated tests verify map resolution logic.
