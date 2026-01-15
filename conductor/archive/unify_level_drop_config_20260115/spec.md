# Track Specification: Unify LevelDrop Configuration

## Overview
The goal of this track is to unify the configuration structure for the `LevelDrop` environment (and potentially others) by moving specific configuration keys (`rewards`, `active_layers`, `action_heads`) into the `env` section of the YAML configuration files. This aligns `level_drop.yaml` with the project's standard configuration schema and ensures these parameters are correctly propagated to the environment and bot logic.

Additionally, the `LevelDropBot` and `LevelDropEnv` classes will be refactored to dynamically utilize these configurations, removing hardcoded logic where possible (specifically for action heads and rewards).

## Functional Requirements

### 1. Configuration Schema Update
*   **Target File:** `config/level_drop.yaml`
*   **Change:** Move the following top-level keys into the `env` section:
    *   `rewards` (and all its sub-keys like `load_reward`, `unload_reward`, etc.)
    *   `active_layers`
    *   `action_heads`

### 2. Environment Refactoring (`LevelDropEnv`)
*   **Dynamic Action Space:**
    *   The `action_space` definition in `LevelDropEnv.__init__` must dynamically calculate its size based on the `action_heads` list provided in the configuration.
    *   Currently, it is hardcoded to `4 * grid_area` (Move, Attack, Load, Unload). It should now be `len(action_heads) * grid_area`.

### 3. Bot Refactoring (`LevelDropBot`)
*   **Dynamic Action Handling:**
    *   `LevelDropBot.apply_action` must process actions based on the index of the `action_heads` list from the config, rather than assuming fixed indices (0=Move, 1=Attack, etc.).
    *   The bot should map the head index (e.g., 0, 1, 2) to the corresponding string (e.g., "move", "attack") and execute the logic for that specific head.
*   **Dynamic Action Masking:**
    *   `LevelDropBot.action_masks` must dynamically construct the mask by iterating over the `action_heads` list.
*   **Reward Calculation:**
    *   Ensure `calculate_drop_reward` explicitly reads values from `self.training_conf.get('env', {}).get('rewards', {})`.

## Acceptance Criteria
*   **Configuration:** `config/level_drop.yaml` has `rewards`, `active_layers`, and `action_heads` nested under `env`.
*   **Functionality:** `LevelDropEnv` initializes correctly with the new config structure.
*   **Behavior:** The bot correctly executes actions (Move, Attack, Load, Unload) based on the order defined in the `action_heads` config.
*   **Verification:** Existing tests for LevelDrop (if any) or a new simple test case verifies that changing the order of `action_heads` in the config changes the bot's behavior/masking structure accordingly.

## Out of Scope
*   Refactoring other levels (Level 7, Level 3) beyond what is strictly necessary to support shared utility changes.
*   Implementing new action heads not already present in the current hardcoded logic.
