# Specification: Level 7 Proximity Reward and Attack Mask Optimization

## Overview
The goal of this track is to improve the training performance of the Level 7 combat agent by introducing a proximity reward (based on optimal unit-to-enemy assignment) and optimizing the attack action mask. This will encourage the agent to approach enemies more effectively and use attack commands more freely.

## Functional Requirements

### 1. Proximity Reward (Optimal Assignment)
- Implement a proximity-based shaping reward in `Level7Bot`, mirroring the logic used in `Level3Bot`.
- Use the `solve_optimal_assignment` utility to pair each allied unit with the most appropriate enemy unit.
- The reward for each unit should be calculated as: `(previous_distance - current_distance) * distance_multiplier`.
- **Distance Multiplier:** `0.05`.
- This reward must be additive to the existing damage and kill rewards.

### 2. Attack Action Mask Optimization
- Modify `Level7Bot.action_masks()` to decouple the Move and Attack masks.
- **Move Mask:** Continue using the existing pathing-based mask (Inverse Mapping of walkable grid).
- **Attack Mask:** Remove the pathing constraint entirely (all `True`), allowing the agent to issue "Attack-Move" commands to any spatial location on the map.

### 3. Reward Configuration
- Update `config/level_7.yaml` to include the `distance_multiplier` and ensure all reward components are correctly balanced.
- **Damage Reward:** Remain at `1.0`.
- **Kill Reward:** Remain at `20.0`.
- **Win Reward:** Remain at `50.0`.
- **Step Penalty:** Remain at `-0.1`.

## Non-Functional Requirements
- **Performance:** The optimal assignment calculation must be efficient enough to not significantly slow down the training loop.
- **Compatibility:** Ensure these changes do not break existing Level 3 or Level 0 logic.

## Acceptance Criteria
- `Level7Bot` correctly calculates and applies the proximity reward during training.
- The attack action mask for Level 7 is unmasked (all valid).
- Unit tests verify that the proximity reward is calculated correctly based on unit movement towards assigned targets.
- A debug training run shows the agent approaching and engaging enemies more proactively than before.

## Out of Scope
- Implementing complex micro-management behaviors (kiting, focus-firing) beyond what the proximity reward encourages.
- Modifying the observation layers or model architecture.
