# Specification: Remove Dilation from Load Action Mask

## Overview
The current `load` action mask in `UnitControlAgent` uses dilation, which artificially expands the valid target area around a unit. This often leads to "ghost targets" where the agent clicks a valid masked cell that is actually too far from the unit center for the `sc2_bot` to execute the load, resulting in a fallback `Move` command. This track will remove dilation from `get_transport_mask` to enforce precise targeting.

## Functional Requirements
1.  **Precise Masking:** The `get_transport_mask` method must NOT use dilation. It should mark only the grid cells that directly correspond to the target unit's position.
2.  **Scope:** This change applies *only* to `get_transport_mask`. Other masks (like terrain navigation) remain unchanged.

## Non-Functional Requirements
1.  **Performance:** Removing the dilation step should slightly improve performance (one less CV2 operation).
2.  **Code Quality:** Maintain existing code style and typing.

## Acceptance Criteria
1.  `get_transport_mask` generates a mask where only the pixels corresponding to the unit's position are True.
2.  No "halo" or expanded area around the unit in the mask.
3.  Existing tests for mask validity still pass (i.e., the mask is not empty).

## Out of Scope
*   Modifying `get_action_mask` (terrain/navigation).
*   Modifying `get_unload_mask`.
*   Changes to `sc2_bot.py` execution logic.
