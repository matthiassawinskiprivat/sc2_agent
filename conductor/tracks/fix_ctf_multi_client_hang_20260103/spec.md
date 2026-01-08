# Specification: Fix CTF Multi-Client Hang

## 1. Overview
The `CaptureTheFlagEnv` hangs during the initialization phase (`_spawn_initial_units`). This is because in a multi-client StarCraft II game, the simulation only advances when *all* connected clients have requested a step (in non-realtime mode). The current implementation only calls `step()` on `bot1.client`, leading to a deadlock where the host waits for the second player to step.

## 2. Functional Requirements
- **Synchronized Stepping:** All calls to `client.step()` during initialization and gameplay must be synchronized across both clients using `asyncio.gather`.
- **Initialization Robustness:** Ensure that debug commands (spawning units, setting health) are correctly applied and followed by synchronized steps.

## 3. Acceptance Criteria
- `CaptureTheFlagEnv` completes `reset()` and moves into the `step()` loop without hanging.
- Initial units (Pylons and units) are spawned correctly for both players.
- The game screen renders the world after the initial steps (resolving the "black screen" hang).
