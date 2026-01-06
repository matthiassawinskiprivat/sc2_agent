# Specification - Fix CTF Single Env Termination

The `CaptureTheFlagSingleEnv` currently lacks the termination logic for when the trainee player loses all their units. This causes the game to continue unnecessarily until the max steps are reached or a pylon is destroyed, leading to inefficient training and incorrect reward attribution.

## Core Requirements

1.  **Termination Condition:** The environment MUST terminate (`done = True`) if the trainee player (Player 1) has no combat-capable units remaining.
2.  **Reward Attribution:** When this termination occurs, the trainee should receive a `loss_penalty` (or appropriate negative reward) as defined in the configuration.
3.  **Parity:** The behavior should match the `CaptureTheFlagEnv` (multi-client) implementation where `trainee_army_lost` triggers termination.

## Validation

*   A reproduction test (or the existing `reproduce_ctf_termination.py` adapted for the SingleEnv class) should confirm that:
    *   Scenario: Trainee has units -> Game continues.
    *   Scenario: Trainee has 0 units -> Game ends immediately with penalty.
    *   Scenario: Opponent has 0 units -> Game continues (God-Mode / Asymmetric logic, or ends if that's the design, but specifically trainee loss is the bug here).
