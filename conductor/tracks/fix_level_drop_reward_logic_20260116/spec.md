# Specification - Fix Level Drop Reward Logic

## Overview
The `LevelDropBot` fails to reward the agent for unloading units near the enemy base because it attempts to find the enemy Command Center in the `self.structures` list (which only contains allied structures). This track fixes the reward calculation to correctly locate the objective using cached positions or the `enemy_structures` list.

## Functional Requirements
- **Correct Objective Lookup:** Modify `calculate_drop_reward` to locate the objective.
    - Primary Method: Use `self._objective_pos` (cached position) to calculate the distance.
    - Fallback Method: If `_objective_pos` is not available, attempt to find the unit in `self.enemy_structures` using `self.command_center_tag`.
- **Reward Trigger:** Ensure the reward is granted when allied units (Marines) are within the specified radius (e.g., 15.0) of the objective position.
- **Fog of War Handling:** The logic must work even if the enemy Command Center is currently in the Fog of War (hence the use of cached position).

## Acceptance Criteria
- `calculate_drop_reward` successfully calculates a positive reward when units are dropped near the enemy Command Center.
- The fix is verified by a test case where the objective is identified but not currently visible (simulated Fog of War) or just by position.
- No regressions in other reward components.

## Out of Scope
- Changing the magnitude of the reward (values remain as configured).
- Modifying the `unload` action mechanics (handled in a separate track).
