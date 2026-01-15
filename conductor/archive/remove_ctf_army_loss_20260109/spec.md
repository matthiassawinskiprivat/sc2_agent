# Specification: Remove Trainee Army Loss Termination Condition in CTF

## Overview
Currently, the Capture the Flag (CTF) environments (Multi-Client, Single-Client, and Self-Play) terminate the episode immediately if the trainee player (Player 1) loses all combat-capable units. This track removes this early termination condition, allowing the game to proceed until a more definitive objective is met (Pylon destruction), a mutual draw occurs, or the time limit is reached.

## Functional Requirements
- **Termination Persistence:** The environment MUST NOT terminate simply because Player 1 (the trainee/host) has no combat units remaining.
- **Revised Termination Conditions:**
    - **Pylon Destruction:** The game ends if either player's Pylon is destroyed (standard Win/Loss).
    - **Mutual Destruction:** The game ends in a draw if BOTH players have no combat units left.
    - **Timeout:** The game ends in a draw if the maximum number of steps (`max_steps`) is reached.
- **Penalty Removal:** The specific `loss_penalty` applied immediately upon the loss of the trainee's army is removed. Rewards will remain focused on Pylon damage/destruction and combat engagements.

## Technical Requirements
- **Multi-Client Environment (`train/envs/capture_the_flag_env.py`):**
    - Modify `_check_termination` to remove the `trainee_army_lost` check.
    - Ensure it returns `done = True` only if both armies are lost or a pylon is dead.
- **Single-Client Environment (`train/envs/capture_the_flag_single_env.py`):**
    - Modify `_async_step` to remove the `p1_army_lost` early termination.
- **Self-Play Environment (`train/envs/capture_the_flag_self_play_env.py`):**
    - Update `_calculate_final_p1_reward` to remove the army-loss terminal penalty.
- **Test Updates:**
    - Update `tests/test_ctf_termination_repro.py` to assert continuation instead of termination when an army is lost.
    - Update `tests/test_ctf_end_conditions_logic.py` to reflect the new logic.

## Acceptance Criteria
1.  **Continuation:** If Player 1 loses all units but their Pylon is alive and Player 2 has units, the game continues.
2.  **Turn Cycling:** When Player 1 has no units, `env.step()` remains functional, effectively skipping P1's actions while P2 continues to act until a terminal state is reached.
3.  **Draw condition:** If both players lose all units, the game terminates with a `draw_penalty`.
4.  **No Regression:** Pylon destruction and timeouts still trigger termination as expected.
