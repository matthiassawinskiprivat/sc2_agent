# Specification: Fix CTF End Conditions and Rewards

## Overview
The 'Capture the Flag' environment currently terminates prematurely or fails to trigger correct end conditions. This track will overhaul the termination logic and reward attribution to ensure games only end when intended, with clear win/loss/draw outcomes.

## Functional Requirements
1.  **Correct End Condition Checks**:
    *   **Win/Loss**: Triggered immediately when a Pylon's health reaches 0.
    *   **Mutual Destruction (Draw)**: Triggered if `player_units[1]` AND `player_units[2]` are both empty, provided both Pylons are still alive.
    *   **Timeout (Draw)**: Triggered when `steps_count` reaches `max_steps`.
2.  **Configurable Rewards**:
    *   Use `win_reward` and a new `loss_penalty` from the configuration file for terminal outcomes.
    *   Implement a `draw_penalty` (e.g., -10.0) for both timeout and mutual destruction scenarios to incentivize decisive play.
3.  **Terminal Reward Logic**:
    *   Ensure that in a Draw, the agent keeps its intermediate rewards (damage/proximity) but receives the `draw_penalty` at the final step.
    *   In a Win/Loss, the terminal reward/penalty is added to the active player's step reward.

## Technical Constraints
*   **Single-Process God-Mode**: Logic must reside in `CaptureTheFlagSingleEnv` and correctly identify unit counts from the bot's unit lists.
*   **Step Persistence**: Verify that `steps_count` is correctly incremented and checked against `max_steps` (which the user noted is set high at 200k, so we must ensure no other truncation logic is interfering).

## Acceptance Criteria
*   Game ends immediately if a Pylon is destroyed.
*   Game ends with a Draw if all mobile units are dead for both sides.
*   Game continues up to `max_steps` if units and pylons are still active.
*   Logs clearly show the reason for termination (Win, Loss, or specific Draw type).
*   Config-driven rewards/penalties are correctly applied to the final step.

## Out of Scope
*   Adding new units or mechanics.
*   Changing the observation space.
