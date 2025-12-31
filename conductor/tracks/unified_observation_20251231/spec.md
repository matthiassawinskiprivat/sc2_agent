# Unified Observation (Guidance for Levels 0-2)

## Context
Currently, `SC2Bot` (used by Levels 0, 1, and 2) sets the `command_objective` layer (Layer 41) to `self.last_target_pos` (the location of the last action). 
However, `Level3Bot` uses this layer to show the **Optimally Assigned Beacon** (the Goal).
This discrepancy causes "Negative Transfer" when moving from Level 2 to Level 3. The agent learns to ignore Layer 41 in early levels (because it's just feedback), making it hard to learn to follow it in Level 3.

## Goal
Update `SC2Bot` to use the **Beacon Position** as the `command_objective`. 
This unifies the meaning of Layer 41 across all levels: "Go Here."

## Strategy
1.  **Modify `SC2Bot._get_observation`**:
    *   Set `raw_obs['command_target']` to `self.beacon.position` (if beacon exists) instead of `self.last_target_pos`.
    *   Ensure `self.beacon` is correctly updated before the observation is generated.

2.  **Verify**:
    *   Create a test `tests/test_unified_observation.py` to confirm `command_target` points to the beacon in `SC2Bot`.

## Detailed Plan
See `plan.md`.
