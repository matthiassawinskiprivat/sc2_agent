# Specification: Fix Persistent Termination Failure in CTF

## Overview
Despite the previous implementation of army detection, the Capture the Flag (CTF) environment continues to ignore army-loss conditions. The game persists until `max_steps` is reached or a Pylon is destroyed, even when one or both players have no visible units remaining on the map. This suggests a fundamental issue with how units are being tracked or filtered within the environment loop.

## Functional Requirements
1.  **Resolve Unit Shadowing:** Investigate and fix potential variable shadowing in `CaptureTheFlagBot`. If the bot is overwriting StarCraft II's internal `units` or `structures` properties with static lists in `__init__`, it will fail to reflect the live game state.
2.  **Comprehensive Army Detection:** 
    *   Expand `_has_combat_units` to be extremely defensive.
    *   Ensure it correctly handles the state where `bot.units` might be empty but the bot still technically owns invisible or technical entities.
3.  **Fix Termination Branching:**
    *   Confirm that `bot1` is correctly identified as the Trainee (Green) and `bot2` as the Opponent (Blue).
    *   Verify that `trainee_army_lost` triggers `done = True`.
    *   Verify that `opponent_army_lost` (Blue) **does not** trigger `done = True` (allowing the Trainee to finish the objective).
4.  **Diagnostic Instrumentation:** Add temporary logging to print the exact tags and types of units currently being detected for both players when the army count is low.

## Acceptance Criteria
-   [ ] The game terminates immediately with a `loss_penalty` when the Trainee (Green) has 0 combat units.
-   [ ] The game continues indefinitely (up to `max_steps`) when the Opponent (Blue) has 0 combat units, provided the Trainee is still alive.
-   [ ] Unit detection is verified to work with the live StarCraft II engine, not just mocks.

## Out of Scope
-   Changes to the combat model or unit stats.
-   Modifications to the reward scale.
