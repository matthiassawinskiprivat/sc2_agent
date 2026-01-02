# Specification: Level 7 Allied Damage and Kill Penalties

## Overview
The goal of this track is to discourage the agent from taking unnecessary damage or losing units by introducing negative rewards (penalties) for allied unit attrition. This will complement the existing offensive rewards and encourage more survival-oriented behaviors.

## Functional Requirements

### 1. Allied Attrition Penalties
- Implement tracking for allied unit health and death within `Level7Bot`.
- **Damage Penalty:** Apply a penalty of `-0.5` for every point of health or shield lost by an allied unit.
- **Kill Penalty:** Apply a penalty of `-10.0` for every allied unit that is destroyed.
- These penalties must be calculated in `Level7Bot` and exposed via a method similar to `calculate_combat_reward`.

### 2. Reward Configuration
- Update `config/level_7.yaml` to include the new penalty parameters:
  - `damage_penalty: -0.5`
  - `kill_penalty: -10.0`
- Ensure these are distinct from the existing `damage_reward` (offensive) and `loss_penalty` (game over).

### 3. Logic Integration
- Modify `Level7Bot` to maintain a `previous_allied_health` dictionary (mapping tags to health + shield).
- Update `Level7Env` to fetch these penalties from the bot and add them to the total step reward.
- The `kill_penalty` and `damage_penalty` should be additive to the existing `loss_penalty` (applied when all units are lost).

## Non-Functional Requirements
- **Efficiency:** The health tracking logic must be performant, as it runs every step.
- **Consistency:** Use the same dictionary-based tracking pattern already implemented for enemy units.

## Acceptance Criteria
- `Level7Bot` correctly identifies and calculates penalties for allied damage and unit loss.
- Unit tests verify that the penalties are correctly summed and applied when allied health decreases or units disappear.
- A debug training run shows the agent receiving negative rewards when taking damage from enemies.
- Configuration changes are correctly loaded from `level_7.yaml`.

## Out of Scope
- Changing the behavior of Level 3 or other environments.
- Implementing healing-based rewards (e.g., for Medivacs or Shield Batteries).
