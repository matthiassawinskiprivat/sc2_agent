# Track Spec: Air Unit Unrestricted Movement and Attack

## Problem Statement
Currently, all units (ground and air) are subjected to an action mask based on the `pathing_grid` in the `UnitControlAgent`. This prevents air units from moving to or attacking targets located over unpathable terrain (e.g., cliffs, voids, or water), even though StarCraft 2 mechanics allow this.

## Objective
Update the action masking logic to detect when the current active unit is an air unit and, in such cases, provide a fully open mask (all actions valid) for both the movement and attack heads.

## Requirements
1. **Air Detection:** Identify if the currently controlled unit has the `is_flying` property set to `True`.
2. **Mask Removal:** If the unit is flying, bypass the `pathing_grid` check and return a mask where all bits are `True`.
3. **Dual-Head Support:** Ensure both the "Move" head and the "Attack" head masks are fully open for air units.
4. **Parity:** Maintain existing ground unit restrictions.
5. **Consistency:** Update both `CTFBot` and `Level7Bot` (and any other relevant bots) to ensure consistent behavior across curriculum levels.

## Constraints
- Do not modify the `UnitControlAgent.get_action_mask` in a way that breaks existing ground unit logic.
- Avoid unnecessary performance overhead in the `action_masks` call.
- The `action_masks` must remain compatible with the `2 * grid_size^2` shape.
