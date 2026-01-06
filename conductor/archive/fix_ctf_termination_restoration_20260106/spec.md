# Specification: Restore Bot Attributes and Fix Final Termination

## Overview
The previous attempt to unshadow unit properties caused an `AttributeError` because the StarCraft II library requires `self.units` and other internal attributes to be present for its `_prepare_step` logic. This track restores those attributes while implementing a more robust termination check inspired by the successful logic in Level 7.

## Functional Requirements
1.  **Restore BotAI Attributes:**
    *   Re-enable `self._initialize_variables()` in `CaptureTheFlagBot.__init__`.
    *   Remove all `delattr` or manual deletions of unit-related properties to prevent `AttributeError`.
2.  **Reliable Army Detection (Level 7 Style):**
    *   Update `_has_combat_units` to use the library's `Units` filtering methods (e.g., `bot.units.exclude_type(...)`).
    *   Ensure the termination check uses the live unit list from the bot instance.
3.  **Asymmetric Termination Logic:**
    *   Confirm `trainee_army_lost` correctly identifies the Trainee (Player 1 / Green).
    *   Ensure that if `trainee_army_lost` is True, the environment returns `done = True` and the `loss_penalty`.
4.  **Fix CTF Single-Env Parity:**
    *   Ensure the same fixes are applied or verified for `CaptureTheFlagSingleEnv` to prevent pipeline crashes.

## Acceptance Criteria
-   [ ] The `train_pipeline` starts and runs without `AttributeError`.
-   [ ] The game terminates immediately with a `loss_penalty` when the Trainee (Green) army is wiped out.
-   [ ] Automated tests verify that `_has_combat_units` returns False when only Larvae/Eggs/Overlords/Workers remain.

## Out of Scope
-   Changing game physics or unit speeds.
-   Modifying the reward function values.
