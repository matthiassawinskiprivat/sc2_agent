# Specification: Dynamic Greedy Reward Optimization

## Overview
Optimize the reward function for multi-unit environments (Level 3 and Level 4) to eliminate "Reward Overlap." By using a globally optimal assignment between units and beacons in every frame, we ensure each unit is rewarded for progress toward its own unique target, encouraging the agent to learn task-splitting and efficient movement rather than grouping up.

## Functional Requirements
- **Globally Optimal Assignment**: In every simulation step where multiple units and beacons exist, calculate the distance matrix between all active units and all available beacons.
- **Dynamic Per-Step Matching**: Use an assignment algorithm (e.g., finding the permutation that minimizes the sum of distances) to pair each unit with a unique beacon *for that frame*.
- **Individualized Distance Reward**: 
    - Reward each unit based on the change in distance to its *assigned* beacon compared to the previous frame.
    - If the number of units exceeds the number of beacons (or vice versa), the matching should still be performed for the subset.
- **Maintain Base Rewards**:
    - **Success Reward**: Large positive reward when a unit consumes any beacon.
    - **Step Penalty**: Small negative reward per step to minimize episode length.

## Non-Functional Requirements
- **Performance**: The matching calculation must be computationally efficient to avoid slowing down the `on_step` game loop, especially as unit counts increase in Level 4.
- **Determinism**: The assignment must be deterministic to provide a stable learning signal.

## Acceptance Criteria
- Level 3 units should exhibit clear splitting behavior, each heading toward a separate beacon when spawned apart.
- The average episode length for Level 3 should decrease as units no longer "hesitate" or group up between targets.
- The implementation must pass tests verifying that rewards are correctly attributed based on the optimal pairing.

## Out of Scope
- Modifications to Level 0, 1, or 2 reward logic.
- Changes to the Observation Transformer or feature layers (handled in previous track).
