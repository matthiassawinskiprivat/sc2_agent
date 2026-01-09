# Specification: CTF Pylon-Only Objective Rendering

## Overview
This track modifies the 'Capture the Flag' (CTF) environment to ensure the `command_objective` observation layer exclusively renders the enemy Pylon's position. This provides a clear, persistent target signal for the agent, preventing distraction from temporary action targets like attacked units.

## Functional Requirements
- **Pylon-Exclusive Rendering:** Only the enemy Pylon's position must be rendered on the `command_objective` layer.
- **Persistent Visibility:** If the enemy Pylon is not currently visible (e.g., in fog of war), the layer should render its last known position.
- **Cross-Agent Visibility:** Both agents in self-play must see the other's Pylon position on their respective objective layers.
- **Remove Distractions:** Movement or attack targets (other than the Pylon) must no longer appear on the `command_objective` layer.

## Technical Requirements
- **Tracking:** Update `CaptureTheFlagBot` to use a dedicated `enemy_pylon_pos` dictionary instead of the generic `command_targets`.
- **Observation:** `get_observation` in `CaptureTheFlagBot` must update the tracked Pylon position whenever visible and pass it as the `command_target` to the transformer.
- **Cleanup:** Remove `command_targets` and stop updating it in `apply_action`.

## Acceptance Criteria
1.  **Objective Persistence:** The `command_objective` layer highlights the enemy Pylon from the start of the episode.
2.  **Fog of War Handling:** If the Pylon is hidden, the marker remains at its last known position.
3.  **Isolation:** Attacking or moving to an enemy unit does NOT cause a marker to appear on the `command_objective` layer.
4.  **Multi-Unit Consistency:** In Multi-Client CTF, both agents correctly see their respective target Pylons.
