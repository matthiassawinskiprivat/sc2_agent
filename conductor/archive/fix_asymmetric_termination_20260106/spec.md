# Specification: Fix Asymmetric Termination Logic

## Overview
The asymmetric termination logic in the Capture the Flag (CTF) environment is currently failing to trigger. Observations show that the game continues even after the Trainee (Player 1 / Green) has lost all mobile units, which prevents the immediate negative feedback loop required for efficient training. Additionally, the condition for the Opponent (Player 2 / Blue) also appears to be ignored or inconsistent.

## Functional Requirements
1.  **Reliable Army Detection:** Improve the detection of "mobile units" to ensure that when a player's combat army is wiped out, the state is correctly identified.
2.  **Restore Asymmetric Termination:**
    *   **Trainee (P1) Army Loss:** MUST trigger `done = True` immediately and apply `loss_penalty`.
    *   **Opponent (P2) Army Loss:** MUST NOT trigger `done = True`. The game should continue to allow the Trainee to destroy the enemy Pylon.
3.  **Entity Filtering:** Ensure that the unit count check correctly ignores non-combat entities (like Pylons) but also remains robust against hidden or "ghost" units that might keep the environment from terminating.

## Acceptance Criteria
-   [ ] The game terminates immediately with a loss reward when the Green player (P1) has 0 combat units.
-   [ ] The game continues when the Blue player (P2) has 0 combat units but their Pylon is still alive.
-   [ ] Automated tests verify that `_check_termination` returns the correct `done` status for various unit-count scenarios.

## Out of Scope
-   Changing the reward values themselves.
-   Modifying the combat mechanics or spawning logic.
