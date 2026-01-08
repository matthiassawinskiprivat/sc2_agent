# Specification: Graceful Termination Handling ("Game has already ended")

## Overview
This track addresses the `sc2.protocol.ProtocolError: ['Game has already ended']` that crashes the training pipeline when a match concludes (e.g., a Pylon is destroyed). The environment needs to detect this state, attribute final rewards correctly, and provide a valid termination signal to the reinforcement learning agent.

## Functional Requirements
1.  **Proactive Error Catching**:
    *   **Environment Level**: `MultiClientEnv.step` must catch `ProtocolError` and check if the underlying cause is an ended game.
    *   **Bot Level**: `CaptureTheFlagBot` must handle cases where game state queries fail because units or pylons no longer exist.
2.  **Final Reward Attribution**:
    *   Implement a robust mechanism to determine the winner (which player's Pylon survived).
    *   Apply the configured `win_reward` to the winner and the equivalent penalty to the loser upon detection of game end.
3.  **Observation Fallback**:
    *   If the game has ended and a new observation cannot be fetched from the engine, return the **last known valid observation** to satisfy the RL agent's step requirement.
4.  **Graceful Loop Termination**:
    *   Ensure `CaptureTheFlagEnv._async_step` returns `done=True` immediately upon detecting game end, preventing further illegal calls to the SC2 client.

## Technical Constraints
*   **SB3 Compatibility**: The `step()` method must always return a valid 5-tuple `(obs, reward, terminated, truncated, info)` even when the engine is unreachable.
*   **Multi-Client Sync**: Both clients must be handled; if one reports 'ended', the entire environment must transition to the terminal state.

## Acceptance Criteria
*   The `train_pipeline` completes a `capture_the_flag` episode and successfully triggers a `reset()` without crashing from `ProtocolError`.
*   The winner receives the full positive reward, and the loser receives the full penalty in the final step.
*   Logs show "Game ended gracefully" instead of critical tracebacks.

## Out of Scope
*   Fixing other SC2 protocol errors unrelated to game termination.
*   Modifying the reward values themselves (only how they are applied).
