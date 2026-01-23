# Implementation Plan - Implement Pickup Guidance

## Strategy
We will modify `LevelDropBot.calculate_drop_reward` to include "Phase 1" meeting rewards. We will maintain per-unit distance tracking for both "Distance to closest ally" and "Distance to objective" to ensure smooth shaping.

## Todo List

### 1. Enhanced State Tracking
- [x] Add `self.previous_meeting_distances = {}` (tag -> float) to `LevelDropBot`.
- [x] Update this in `update_reward_states` based on the closest relevant partner (Transport <-> Infantry).

### 2. Implement Meeting Rewards
- [x] Modify `LevelDropBot.calculate_drop_reward` to:
    - If actor is an **empty transport**: Calculate shaping based on distance to closest allied infantry.
    - If actor is an **infantry unit**: Calculate shaping based on distance to closest allied transport.
    - Use a separate `meeting_reward_scale` (or reuse `distance_reward_scale`) for these signals.

### 3. Synchronize Phase Transitions
- [x] Ensure that as soon as `cargo_used > 0`, the transport switches entirely to the "Objective" distance reward.
- [x] Ensure infantry units stop getting meeting rewards once they are within a very small distance (e.g., 2.0) or after they are loaded.

### 4. Verification
- [x] Update `tests/test_drop_reward_logic.py` to verify:
    - Empty Medivac gets rewarded for moving toward a Marine.
    - Marine gets rewarded for moving toward a Medivac.
    - Loaded Medivac switches to Objective reward.
