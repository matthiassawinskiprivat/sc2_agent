# Specification: Asymmetric Self-Play for Capture the Flag

## Overview
This track implements an asymmetric training mode for the Capture the Flag (CTF) environment. Instead of alternating control between both players, the environment will focus training on a single player (Player 1) while the opponent (Player 2) is controlled by a snapshot of the current model. This improves training efficiency by removing noise from "dead player" turns and creating a tighter feedback loop for combat and objective-based learning.

## Functional Requirements
1.  **Fixed Trainee Role:** Player 1 is designated as the primary trainee. Training data and rewards are collected only for Player 1.
2.  **Automated Opponent:** Player 2 actions are automatically generated using the latest model weights, snapshotted at the start of each episode (`reset`).
3.  **Early Termination (Trainee Loss):** If Player 1 (the trainee) loses all mobile units, the episode ends immediately as a loss.
4.  **Persistent Objective (Opponent Loss):** If Player 2 loses all mobile units, the episode **continues**. This allows Player 1 to learn the secondary objective of navigating to and destroying the enemy Pylon.
5.  **Reward Logic:**
    *   When Player 1 loses all units: Apply the `loss_penalty` from the configuration.
    *   Standard Win/Loss rewards apply upon Pylon destruction or timeout.
6.  **Perspective Consistency:** Ensure `PerspectiveTransformer` correctly flips observations and actions for the automated Player 2 so it behaves as a valid adversary.

## Non-Functional Requirements
-   **Training Efficiency:** Reduce the number of "empty" environment steps processed by the trainer.
-   **Stability:** Use snapshot-based self-play (reset-level updates) to prevent high-frequency feedback loops during a single episode.

## Acceptance Criteria
-   [ ] The environment terminates immediately if Player 1 has 0 units.
-   [ ] The environment continues if Player 2 has 0 units but Player 1 still has units.
-   [ ] Player 2 actions are generated automatically without requiring input from the external `step(action)` call.
-   [ ] The `loss_penalty` is correctly applied to the reward signal upon early termination.
-   [ ] Observations for the trainer always correspond to Player 1.

## Out of Scope
-   Training Player 2 (it only uses inference).
-   Dynamic switching of the trainee ID during a single training run (fixed to P1 for now).
