# Specification - Fix LevelDrop Initialization and Objective Logic

## Overview
The `LevelDrop` environment currently behaves like `Level0Env`, clearing pre-placed units and spawning irrelevant ones (Zergling/Beacon). This track fixes the environment registration, prevents unit destruction on startup, enables multi-unit-type control (Marines + Medivacs), and implements robust objective tracking for enemy main structures across all races.

## Functional Requirements
- **Environment Correction:** Update `level_drop_0.yaml` to use `env_class: "LevelDropEnv"`.
- **Preserve Map State:** Ensure `LevelDropBot` does not kill existing units or structures during initialization.
- **Multi-Unit Control:** 
    - `LevelDropBot` must control both `MARINE` and `MEDIVAC` unit types.
    - The environment must cycle through all active units (Marines + Medivac) for action assignment.
- **Dynamic Objective Tracking:**
    - Automatically identify the enemy main structure (`COMMANDCENTER`, `HATCHERY`, or `NEXUS`).
    - Render the objective's position on the `command_objective` layer.
    - **Intelligence Protocol:** The objective position must be rendered on the observation layer even if the structure is currently hidden by Fog of War.

## Technical Details
- **`config/level_drop_0.yaml`:** Change `name: "LevelDrop"` to `env_class: "LevelDropEnv"`.
- **`LevelDropBot`:**
    - Override `reset_game_state` to remove cleanup logic.
    - Override `get_active_units` to return all units matching Marines or Medivacs.
    - Implement cross-race enemy base detection logic.
- **Objective Layer:** Ensure `raw_obs['command_target']` is populated with the enemy base position during every observation generation.

## Acceptance Criteria
- Starting the environment does NOT spawn a Zergling/Beacon.
- Marines and Medivac remain on the map as placed in the SC2 editor.
- The agent successfully receives observations for both Marines and the Medivac in sequence.
- The `command_objective` layer correctly highlights the enemy base, even when unscouted.
