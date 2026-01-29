# Specification - Fix Cross-Race Spatial Unload Mechanics

## Overview
The `unload` action in `SC2Bot` is currently non-functional for many units (like Medivacs) and ignores the agent's spatial intent. This track fixes the implementation to correctly map the `unload` action to race-specific "Unload All At" abilities using the spatial `target_pos` provided by the agent.

## Functional Requirements
- **Race-Specific Command Mapping:** Map the `unload` action head to the correct unit-specific ability:
    - Terran Medivac: `AbilityId.UNLOADALLAT_MEDIVAC`
    - Zerg Overlord: `AbilityId.UNLOADALLAT_OVERLORD`
    - Protoss Warp Prism: `AbilityId.UNLOADALLAT_WARPPRISM`
- **Spatial Targeting:** Pass the `target_pos` (derived from the spatial action index) to the unload command.
- **Pre-condition Check:** Maintain the `unit.cargo_used > 0` check before issuing the command.
- **Fallback Logic:** Provide a fallback to `AbilityId.UNLOADALLAT` or `AbilityId.UNLOADALL` for unknown transport units.

## Acceptance Criteria
- `SC2Bot.apply_action` correctly identifies the transport unit type and selects the corresponding `AbilityId`.
- The `unload` command is issued with the `target_pos` coordinate.
- Automated tests verify that for a Medivac, the `UNLOADALLAT_MEDIVAC` ability is called with the expected coordinates.
- The system remains stable for Terran, Zerg, and Protoss transport units.

## Out of Scope
- Fixing the reward logic in `LevelDropBot` (the "enemy structure in allied list" bug).
- Handling "Unload Unit" (single passenger) actions.
- Researching upgrades (e.g., Medivac speed) within this track.
