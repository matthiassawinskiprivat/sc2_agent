# Specification: Improve Loading Logic with AbilityIds

## Overview
The current "load" implementation in `sc2_bot.py` relies solely on the `unit.smart()` command. This track will upgrade the logic to use explicit `AbilityId` commands, supporting both units entering transports and transports picking up units across all three races (Terran, Zerg, Protoss).

## Functional Requirements
1.  **Dual-Mode Loading:**
    *   **Unit-Centric:** If a non-transport unit (e.g., Marine) is controlled, it should use `AbilityId.SMART` on a nearby transport.
    *   **Transport-Centric:** If a transport unit (Medivac, Overlord, Warp Prism) is controlled, it should use a specific `LOAD` ability on the target unit.
2.  **Race-Specific Mapping:**
    *   `UnitTypeId.MEDIVAC` -> `AbilityId.LOAD_MEDIVAC`
    *   `UnitTypeId.OVERLORD` -> `AbilityId.LOAD_OVERLORD`
    *   `UnitTypeId.WARPPRISM` / `UnitTypeId.WARPPRISMPHASING` -> `AbilityId.LOAD_WARPPRISM`
3.  **Robust Targeting:**
    *   The bot should identify a valid target (either a transport or a loadable unit) within a reasonable radius of the `target_pos`.
4.  **Fallback Mechanism:**
    *   If the unit is not a transport or no specific mapping is found, fall back to `AbilityId.SMART`.

## Non-Functional Requirements
1.  **Code Quality:** Adhere to existing `python-sc2` and project conventions.
2.  **Maintainability:** Ensure the mapping is easily extendable for other transport types (e.g., Bunkers, Nydus Worms) in the future.

## Acceptance Criteria
1.  Marines can successfully enter a nearby Medivac using the `load` action.
2.  Medivacs can successfully pick up a nearby Marine using the `load` action.
3.  The logic works correctly for Overlords (Zerg) and Warp Prisms (Protoss).
4.  No regressions in existing "move" or "attack" logic.
5.  Automated tests verify both "enter" and "pickup" scenarios.

## Out of Scope
*   Implementing complex "multi-pickup" pathing (picking up multiple spread-out units in one command).
*   Handling unload logic (already exists, though may be touched if refactoring is needed).
