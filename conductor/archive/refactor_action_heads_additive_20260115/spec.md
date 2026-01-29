# Track Specification: Refactor Action Heads Configuration

## Overview
The goal of this track is to refactor the `action_heads` configuration mechanism to be additive and dynamic, similar to `active_layers`. This ensures that as agents progress through the training curriculum (from Level 0 to Level 7 to Level Drop), they automatically inherit action capabilities from previous levels.

We will implement this logic in `ConfigManager` and refactor the environments and bots for Level 0, Level 7, and Level Drop to utilize this dynamic configuration.

## Functional Requirements

### 1. Centralized Configuration Logic (`ConfigManager`)
*   **New Method:** Implement `get_cumulative_action_heads(target_level)` in `src/utils/config_manager.py`.
*   **Logic:**
    *   Iterate through levels defined in `pipeline.yaml`.
    *   Accumulate `action_heads` defined in each level's `env` configuration.
    *   **Off-Pipeline Support:** If `target_level` (e.g., `level_drop`) is not in `pipeline.yaml`, assume it follows the **entire** pipeline sequence and accumulate all previous heads before adding its own.

### 2. Configuration Updates
*   **`config/level_0.yaml`:** Add `action_heads: ["move"]` to the `env` section.
*   **`config/level_7.yaml`:** Add `action_heads: ["attack"]` to the `env` section.
*   **`config/level_drop.yaml`:** Update `action_heads` to only contain `["load", "unload"]` (removing "move" and "attack" to test accumulation).

### 3. Code Refactoring (Environments & Bots)
*   **`Level0Env` / `Level0Bot`:** Refactor to initialize action space and masks based on `ConfigManager.get_cumulative_action_heads`.
*   **`Level7Env` / `Level7Bot`:** Refactor to initialize action space and masks based on `ConfigManager.get_cumulative_action_heads`.
*   **`LevelDropEnv` / `LevelDropBot`:** Ensure they continue to work with the new additive logic (verifying "move", "attack", "load", "unload" are present).

## Acceptance Criteria
*   `ConfigManager` correctly returns `['move']` for Level 0.
*   `ConfigManager` correctly returns `['move', 'attack']` for Level 7.
*   `ConfigManager` correctly returns `['move', 'attack', ..., 'load', 'unload']` for Level Drop (assuming it inherits from the full pipeline).
*   Environments initialize with the correct action space dimensions.
*   Bots generate valid action masks corresponding to the cumulative heads.

## Out of Scope
*   Modifying `pipeline.yaml` structure beyond reading it.
*   Changing the actual behavior of the actions (just the wiring/configuration).
