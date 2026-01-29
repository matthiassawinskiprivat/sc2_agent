# Specification - Implement Action Masking for Load Head

## Overview
Currently, the `load` action head in `SC2Bot` is unmasked, allowing the agent to target any pixel on the map. This creates sparse targets and noise during training. This track implements a spatial mask for the `load` head that only allows targeting areas near valid transport units (Medivacs, Overlords, Warp Prisms).

## Functional Requirements
- **Spatial Masking for Load:** Implement a method to generate a boolean mask for the `load` action head.
- **Transport Identification:** The mask must identify all allied transport units currently on the map.
- **Dilation/Radius:** The mask should include a small radius (dilation) around each transport unit to ensure the agent can reliably "hit" the target.
- **Agent Integration:** Integrate this mask into `UnitControlAgent.get_action_mask` or a similar method so `SC2Bot.action_masks` can retrieve it.

## Technical Details
- **Location:** Implementation will primarily occur in `src/agent/unit_control_agent.py`.
- **Logic:** 
    1. Filter allied units for transport types.
    2. Convert their world positions to grid coordinates.
    3. Create a binary grid and apply a kernel-based dilation (similar to the movement mask).
    4. Pass this mask back to `SC2Bot`.

## Acceptance Criteria
- `UnitControlAgent` can generate a mask specifically for transport units.
- The `load` head in `SC2Bot.action_masks()` uses this specific mask instead of the `unrestricted_mask`.
- Unit tests verify that the mask is `True` near transport units and `False` elsewhere.
- Training performance (discovery of the load mechanic) is theoretically improved.

## Out of Scope
- Implementing masking for other non-spatial heads (if any).
- Modifying the underlying CNN architecture.
