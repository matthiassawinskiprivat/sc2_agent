# Specification - Fix LevelDrop Termination Logic

## Overview
The `LevelDropEnv` currently suffers from "Instant Victory" syndrome. Because it inherits from `Level7Env`, it checks for a specific enemy unit type (defaulting to ZERGLING). Since the custom drop map uses structures (Command Center) and potentially different unit types, the environment finds 0 "target" units and terminates immediately. This track implements a "Total Annihilation" condition required for the drop mission.

## Functional Requirements
- **Total Annihilation Win Condition:** The episode must only end in victory when BOTH the enemy's units and structures are completely destroyed (count = 0).
- **Defeat Condition:** The episode ends in failure if the agent loses all controlled units (Marines and Medivacs).
- **Persistence:** The environment must remain active as long as any enemy structure or unit remains on the map, allowing the agent time to perform the drop and attack sequence.

## Technical Details
- **`LevelDropEnv`**: Override the `_async_step` method to provide specialized termination logic.
- **Entity Counting:** Utilize `self._bot.enemy_units` and `self._bot.enemy_structures` to verify the total count of enemy entities.
- **Controlled Unit Tracking:** Use the `LevelDropBot.get_active_units()` list to determine if the agent still has units to play with.

## Acceptance Criteria
- `LevelDropEnv` no longer terminates immediately upon reset.
- Episode concludes only after the enemy main building (and any defending units) are removed from the map.
- Correct win/loss rewards are applied at the moment of termination.

