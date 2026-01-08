# Specification: Speed Up CTF Environment Start-Up

## Overview
This track aims to reduce the time required to initialize the `CaptureTheFlagEnv`. Currently, the start-up process is perceived as slow. The goal is to optimize the handshake and game setup phases by minimizing redundant delays while strictly maintaining the current system stability.

## Functional Requirements
1.  **Audit Stability Delays**:
    *   Identify all `asyncio.sleep` or hardcoded timeout values in `MultiClientSC2Manager`, `MultiClientEnv`, and `CaptureTheFlagEnv`.
    *   Experimentally determine the minimum required delay for each step without introducing "Connection already closed" or protocol handshake failures.
2.  **Optimize Process Handshake**:
    *   Review the `launch_processes` logic in `MultiClientSC2Manager` to ensure SC2 instances are ready for connection as quickly as possible.
3.  **Streamline Game Setup**:
    *   Audit the sequence of `create_game`, `join_game`, and initial `_synchronized_step` calls in `CaptureTheFlagEnv` to eliminate unnecessary waiting.
4.  **Stability Guardrails**:
    *   Every reduction in delay must be verified against a stress test (multiple consecutive resets) to ensure zero regression in stability.

## Technical Constraints
*   **Stability First**: Optimization must not increase the frequency of `ConnectionAlreadyClosed` or `ProtocolError` during initialization.
*   **Compatibility**: Must remain compatible with Windows and the current version of `burnysc2`.

## Acceptance Criteria
*   Start-up time (from `reset()` call to first `step()`) is significantly reduced (target: >20% improvement).
*   Zero initialization failures in 50 consecutive `reset()` calls.
*   Logs clearly state the new optimized delay values and the reasoning/testing behind them.

## Out of Scope
*   Re-implementing SC2 process persistence (process reuse across resets).
*   Batching debug commands (unless a specific bottleneck is found there).
*   Optimizing the internal SC2 map loading engine itself.
