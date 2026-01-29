# Specification - Restore Game Creation with Dynamic Race Detection

## Overview
The `LevelDropEnv` currently fails with a `ProtocolError` because it attempts to join a game that hasn't been created. This track restores the `create_game` lifecycle and implements a dynamic race detection system that reads the agent and enemy races directly from the map filename.

## Functional Requirements
- **Lifecycle Restoration:** Re-introduce the `create_game` call in `LevelDropEnv._setup_game_session`.
- **Dynamic Race Parsing:** 
    - Parse the agent's race and the computer's race from the map name using the `_vs_` delimiter.
    - Example: `...Terran_vs_Zerg` -> Agent: `Race.Terran`, Enemy: `Race.Zerg`.
- **Preserve Units:** Ensure that `create_game` combined with the existing `LevelDropBot.reset_game_state` (which skips cleanup) correctly preserves the Marines and Medivacs placed in the SC2 editor.

## Technical Details
- **`LevelDropEnv`**:
    - Implement `_parse_races_from_map_name()` helper logic.
    - Update `_setup_game_session` to call `self._controller.create_game` with the parsed races.
- **Race Mapping**: Support `Terran`, `Zerg`, and `Protoss` string-to-Race conversions.

## Acceptance Criteria
- `train_pipeline.py` starts without `ProtocolError`.
- The game is created with the races specified in the map name.
- Pre-placed units (8 Marines + 1 Medivac) are present upon game start.
- `command_objective` layer correctly identifies the enemy main structure corresponding to the parsed enemy race.
