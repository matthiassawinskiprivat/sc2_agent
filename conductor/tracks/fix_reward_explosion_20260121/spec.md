# Track Spec: Fix Reward Explosion & Attribution

## Goal
Normalize reward scaling and fix the multiplication of event-based rewards (loading/unloading) to ensure stable and meaningful reinforcement signals.

## Requirements
1. **Calibrate Damage Rewards:**
    - Adjust `damage_reward` in `level_drop_0.yaml` from `1.5` to a more sensible value (e.g., `0.01` or `0.05`).
    - The total reward for destroying the main objective should be comparable to or slightly higher than the `win_reward`.

2. **Strict Event Attribution:**
    - `load_reward` and `unload_reward` (arrival reward) must only be awarded to the specific unit(s) involved in the event during their specific turn in the cycle.
    - Prevent other units in the same cycle from "leaking" these rewards.

3. **Granular State Tracking:**
    - Implement per-unit state tracking for cargo and passenger status to enable precise attribution without relying on global cycle-end updates for discrete events.
