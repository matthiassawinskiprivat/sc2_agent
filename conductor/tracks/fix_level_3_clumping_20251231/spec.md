# Fix Level 3 Agent Clumping (Observation & Guidance)

## Context
The Level 3 agent (multi-unit control) is "clumping" units together instead of splitting them to cover multiple beacons. Although the "Dynamic Greedy Reward" correctly incentives splitting, the agent physically cannot see the alternative targets because the observation logic filters beacons to only show the "closest" one. Additionally, the agent lacks explicit visual guidance on *which* beacon is its assigned target.

## Goal
Enable the agent to learn splitting behavior by:
1.  **Removing Observation Blindness:** Allowing the agent to see ALL beacons in the `enemy_unit_density` layer.
2.  **Explicit Visual Guidance:** Using the `command_objective` layer (Layer 41) to highlight the *optimally assigned* beacon for the specific unit being controlled.

## Strategy
1.  **Modify `Level3Bot.get_observation_for_unit`**:
    *   Pass `list(beacons)` instead of `[closest_beacon]` to `raw_obs['enemy_units']`.
    *   Lookup the unit's assigned target from `self.current_assignments`.
    *   Set `raw_obs['command_target']` to the position of that assigned beacon.

2.  **Verify Observation**:
    *   Use `debug_visuals.py` or a new test to confirm the agent sees all beacons and the crosshair appears on the correct one.

3.  **Train & Verify**:
    *   Run a short training session to confirm the agent starts splitting.

## Detailed Plan
See `plan.md` for the step-by-step execution plan.
