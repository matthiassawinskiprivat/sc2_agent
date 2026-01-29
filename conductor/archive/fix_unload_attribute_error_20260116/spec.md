# Specification - Fix Unload AttributeError in SC2Bot

## Overview
The `LevelDropEnv` training fails when the agent attempts to unload units from a Medivac. The `SC2Bot.apply_action` method calls a non-existent method `unload_all_at` on the `Unit` object, leading to an `AttributeError`. This track fixes the error by implementing the standard "Unload All" command using the correct `burnysc2` API.

## Functional Requirements
- **Standard Unload Implementation:** Replace the invalid `unit.unload_all_at(target_pos)` call with a command that unloads all units at the transport's current position.
- **Correct API Usage:** Utilize the `AbilityId.UNLOADALL` to trigger the unload action.
- **Pre-condition Check:** Maintain the check for `unit.cargo_used > 0` before issuing the unload command to avoid redundant actions.

## Technical Details
- **`SC2Bot.apply_action`**: 
    - Update the `unload` action branch.
    - Import `AbilityId` from `sc2.ids.ability_id`.
    - Change logic to `cmd = unit(AbilityId.UNLOADALL)`.

## Acceptance Criteria
- `train_pipeline.py` no longer crashes when the agent attempts to unload.
- Medivacs correctly unload all passengers when the "unload" action is triggered.
- Integration tests confirm the fix.

