# Plan: Fix Unload Action Handler

**Goal**: Fix the silent failure of the 'unload' action in `SC2Bot.apply_action` by adding a missing handler for the spatial execution path, ensuring robustness against configuration mismatches.

## Context
The user transitioned the 'unload' action from a spatial head to a categorical head. However, the agent successfully loads units and flies to the target but fails to unload. Code analysis revealed that `apply_action` in `src/sc2_bot.py` has a handler for `unload` in the **categorical** block but is missing one in the **spatial** block. If the configuration still lists `unload` in `action_heads` (spatial), or if the index resolution maps it there, it fails silently.

## Strategy
1.  **Robustness Update**: Modify `src/sc2_bot.py`.
2.  **Spatial Handler**: Add an `elif action_name == "unload":` block to the spatial section of `apply_action`.
    *   *Implementation*: If `unload` is triggered spatially (with a target coordinate), the bot should order the unit to unload at that target (or move there and unload).
    *   *Logic*: Use `AbilityId.UNLOADALL_AT` if available, or `smart` on the ground? actually `UNLOADALL` is usually immediate.
    *   *Fallback*: If the unit is a transport, `cmd = unit(AbilityId.UNLOADALL_AT, target_pos)` might work for Medivacs/WarpPrisms if targeting the ground.
    *   *Verification*: Check `sc2` python library capabilities for `UNLOADALL_AT`. Medivacs have `MEDIVACUNLOADALL_AT`.
3.  **Safety Logging**: Add a final `else` in the spatial block to log an error if an unknown action name is encountered, preventing future silent failures.

## Todo List
- [x] Update `src/sc2_bot.py` to include the spatial `unload` handler.
- [x] Verify `sc2` python library constants for Unload At Target.