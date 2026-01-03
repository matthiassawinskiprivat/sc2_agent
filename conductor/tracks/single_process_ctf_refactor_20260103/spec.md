# Specification: Single-Process CTF Refactoring

## Overview
Refactor the `CaptureTheFlagEnv` from a multi-client (two-process) architecture to a single-process architecture. This will fix the inconsistency where `num_envs: 1` launched two game instances, simplify state synchronization, and reduce system resource consumption. The environment will use a single "God-Mode" connection to control units for both players.

## Functional Requirements
1.  **Single-Process Transition**:
    *   Deprecate the use of `MultiClientEnv` and `MultiClientSC2Manager` for Capture the Flag.
    *   Implement a new `CaptureTheFlagEnv` that inherits from a single-client base or manages a single `SC2Process` directly.
    *   Ensure that `num_envs: N` in the training configuration results in exactly `N` SC2 processes.
2.  **"God-Mode" Unit Control**:
    *   The single bot connection must be able to control units belonging to both Player 1 and Player 2.
    *   This will likely involve using `debug_create_unit` to spawn units for different owners and issuing commands via the single player's client.
3.  **Perspective Handling (Unit-Only Flip)**:
    *   Maintain the turn-based "Self-Play" cycle.
    *   When it is Player 2's turn to act (or for the self-play model to act):
        *   Swap the "Allied" and "Enemy" labels in the observation layers.
        *   Do NOT rotate map coordinates or unit positions.
4.  **Reward and Termination Logic**:
    *   Update the reward calculation and game-end detection to work within a single-client state.
    *   Ensure the winning/losing pylon detection correctly identifies which "side" was defeated.

## Technical Constraints
*   **SC2 Debug Interface**: Must utilize the SC2 debug commands (e.g., `debug_create_unit`, `debug_set_unit_value`) to manage units for both sides through a single connection.
*   **State Consistency**: The game state must be the "single source of truth," eliminating synchronization issues between two processes.

## Acceptance Criteria
*   Running training with `num_envs: 1` launches exactly one SC2 window.
*   The agent can successfully control units and receive rewards on its turn.
*   The self-play model can control the "opponent" units on its turn.
*   A 10-cycle reset test confirms 100% stability in the new single-process setup.
*   Training pipeline runs for at least 50,000 steps without crashing or process leaks.

## Out of Scope
*   Rotating coordinates/layers (Map Rotation).
*   Supporting 3+ players.
*   Optimizing the CNN architecture itself.
