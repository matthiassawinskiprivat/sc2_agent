# Specification: Fix AttributeError and Persistent Termination Failure

## Overview
The removal of `_initialize_variables()` in the previous fix caused a critical `AttributeError` during training startup because the StarCraft II library requires several internal variables (like `distance_calculation_method`) to be initialized. This track aims to restore library compatibility while simultaneously solving the persistent termination failure where games continue after army loss.

## Functional Requirements
1.  **Restore BotAI Compatibility:**
    *   Restore the call to `self._initialize_variables()` in `CaptureTheFlagBot.__init__`.
    *   Immediately after the call, explicitly `del` or clear any attributes that shadow the library's dynamic properties (like `units`, `structures`, `all_units`, `larva`) to ensure the bot uses the library's live getters.
2.  **Robust Army Detection:**
    *   Update `_has_combat_units` to use `type_id` for filtering.
    *   Ignore all non-combat entities: `LARVA`, `EGG`, `OVERLORD`, `PROBE`, `SCV`, `DRONE`.
    *   Ensure the check specifically looks for units owned by the player.
3.  **Environment Sanity Check:**
    *   Add a check in the environment's `reset()` method to log and verify that unit properties are working as expected (i.e., not static lists).
4.  **Fix Asymmetric Logic:**
    *   Verify that `bot1` (Green) army loss ends the game.
    *   Verify that `bot2` (Blue) army loss allows the game to continue for the objective.

## Acceptance Criteria
-   [ ] The `train_pipeline` starts without `AttributeError`.
-   [ ] The game terminates immediately with a `loss_penalty` when the Trainee (Green) has 0 combat units.
-   [ ] Diagnostic logs in `reset()` confirm that `bot.units` is a dynamic property, not a shadowed list.
-   [ ] Automated tests verify the `_has_combat_units` logic using `type_id`.

## Out of Scope
-   Changes to non-CTF bots.
-   Modifying reward scaling.
