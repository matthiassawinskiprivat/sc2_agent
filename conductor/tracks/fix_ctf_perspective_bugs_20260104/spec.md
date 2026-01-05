# Specification - Fix CTF Perspective Reward and Target Bugs

## Overview
Agents in CTF exhibit a right-edge bias due to non-perspective-aware logic in `CaptureTheFlagBot`.

## Issues
1. **Reward Mismatch:** `calculate_reward` uses `self.enemy_structures` (fixed to connection owner) instead of perspective-flipped structures. This rewards Player 2 for moving right (towards its own pylon).
2. **Target Collision:** `self.command_target` is shared, causing players to see each other's last targets as their own objective.

## Functional Requirements
* **Flipped Structure Access:** Use `self.structures` and `self.enemy_structures` through a flipping logic in `calculate_reward`.
* **Per-Player State:** Use `self.command_targets = {1: None, 2: None}` and `self.prev_dist_to_pylon = {1: None, 2: None}`.
* **Observation Logic:** `get_observation` must pull from the correct `command_targets[p_id]`.

## Acceptance Criteria
* Symmetric movement behavior verified in Single-Client God-Mode.
* Objective layer correctly shows enemy pylon for both players.
