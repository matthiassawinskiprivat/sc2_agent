# Track: Medivac Heal Action Masking

## Context
The Medivac uses the 'attack' head for healing. However, when no units need healing, issuing an 'attack' command is redundant with 'move'. Masking the 'attack' head when no damaged biological allies are in range will reduce action space noise and help the agent learn the specific utility of the 'attack' head as a 'heal' tool.

## Goals
-   Mask the entire 'attack' action head for Medivacs when no damaged biological allied units are in healing range.
-   When a valid heal target is in range, allow the 'attack' head to use the standard `move_mask` (unrestricted air movement).
-   Ensure this does not affect other units (Marines should still have their 'attack' head active).

## Requirements
1.  **Medivac Attack Logic:**
    *   Identify if the active unit is a Medivac.
    *   Scan all allied units for those that are `is_biological` and `health < health_max`.
    *   Calculate distance from Medivac to these units.
    *   If `min(distance) > HEAL_RANGE + medivac_radius + target_radius`: Mask the entire 'attack' head.
    *   Otherwise: Use `move_mask`.
2.  **Implementation:**
    *   Modify `src/sc2_bot.py` to handle the conditional head masking.
    *   Potentially add a `get_heal_mask` or similar helper if needed, though simple logic in `action_masks` might suffice.

## Non-Goals
-   Changing how the 'attack' command is executed in `apply_action`.
-   Masking enemy units (this is a spatial mask, and we want to allow Attack-Move near units).
