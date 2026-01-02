# Specification: Greedy Closest-Target Logic for Level 3 and Level 7

## Overview
Change the target assignment logic in `Level3Bot` and `Level7Bot` from an optimal assignment (Hungarian algorithm) to a greedy "closest target" approach. This applies to both the visual observation (the `command_target` layer) and the distance-based shaping reward.

## Functional Requirements

### 1. Closest Target Logic
- Replace `_calculate_optimal_assignment` and `_calculate_optimal_assignment_combat` with logic that identifies the closest target for each individual unit.
- Multiple units are allowed (and expected) to target the same closest target if it is the nearest for all of them.

### 2. Observation Update
- In `get_observation_for_unit`, the `command_target` provided to the `HierarchicalAgentSystem` must be the position of the unit's **closest** target.

### 3. Reward Calculation
- The distance-based reward will be a shaping reward based on the **change in the average distance** of all active units to their respective closest targets.
- `Reward = (avg_previous_distances - avg_current_distances) * multiplier + step_penalty`.

### 4. Scope
- **Level 3:** Units (Zerglings) target Beacons.
- **Level 7:** Allied units (e.g., Marines) target Enemy units (e.g., Zerglings).
- **Combat Logic:** Damage and kill rewards in Level 7 remain independent and unchanged.

## Non-Functional Requirements
- **Performance:** Using closest target ($O(N \times M)$) is more computationally efficient than optimal assignment ($O(N^3)$), which should slightly improve training step speed.
- **Maintainability:** Remove or deprecate the optimal assignment helper calls within these specific bots to simplify the logic.

## Acceptance Criteria
- [ ] Units' observations reflect the closest target.
- [ ] Reward is correctly calculated based on the average progress toward closest targets.
- [ ] Multiple units successfully converge on the same target if it is the closest for both.
- [ ] All existing tests for Level 3 and Level 7 continue to pass (or are updated if they strictly enforced optimal assignment).

