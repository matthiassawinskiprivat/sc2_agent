# Specification: Individual Unit Reward Refactoring

## Overview
This track aims to improve the training efficiency and behavior of the agent in multi-unit environments (Level 3, 4, and 7). The current implementation uses an averaged reward across all units, delivered only at the end of a game step cycle. This dilutes the reinforcement signal and leads to "centroid attraction," where units fail to commit to individual targets. We will refactor the logic to provide immediate, individual rewards for each unit's action.

## Functional Requirements
- **Individual Distance Shaping:** In `Level3Bot` and `Level7Bot`, calculate the change in distance for the specific unit currently taking an action, rather than averaging the change across all units.
- **Per-Action Step Penalty:** Apply the configured step penalty to every individual unit action within the cycling loop in `Level3Env`.
- **Targeted Success Rewards:**
    - In Level 3/4, the `success` reward for collecting a beacon must be attributed specifically to the unit that triggered the collection.
    - In Level 7, rewards for damage dealt or kills should be attributed to the unit taking the action (if it's an attack action) or handled via individual distance rewards.
- **Environment Loop Update:** Modify `Level3Env._async_step` (and any related logic in Level 7) to return these individual rewards immediately after each unit's action, instead of returning `0.0` for intermediate steps.

## Non-Functional Requirements
- **Credit Assignment:** Ensure that the PPO algorithm receives a clear signal for every transition in the unit-cycling buffer.
- **Consistency:** Maintain the same reward scales (multipliers) defined in the existing YAML configurations, but apply them per-unit.

## Acceptance Criteria
- [ ] Units in Level 3/4 show a higher tendency to move directly to their assigned targets rather than clumping or staying in the middle.
- [ ] The `_async_step` function in `Level3Env` returns a non-zero reward for intermediate unit actions if they make progress or hit a success condition.
- [ ] Unit-specific progress is accurately tracked in `Level3Bot.previous_distances` and `Level7Bot.previous_distances`.
- [ ] All existing tests pass, and new tests verify the individual reward attribution.

## Out of Scope
- Changing the greedy target assignment algorithm itself (this track focuses on the reward feedback for the existing assignments).
- Refactoring Level 0 or Level 1 (single-unit environments).
