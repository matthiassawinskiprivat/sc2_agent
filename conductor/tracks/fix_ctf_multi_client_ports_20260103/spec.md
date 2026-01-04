# Specification: Fix CTF Multi-Client Ports

## 1. Overview
This track addresses an `EOFError` and `BrokenPipeError` that occur during the `reset()` phase of the `CaptureTheFlagEnv`. These errors are caused by the lack of `Portconfig` when launching multiple SC2 clients, which prevents them from communicating with each other and results in a crash of the worker process.

## 2. Functional Requirements
- **Logging Integration:** Add standardized logging to `MultiClientEnv` (matching `Level0Env`) to ensure that any future crashes in worker subprocesses are captured in `logs/worker_*.log`.
- **Port Allocation:** Implement `Portconfig` in `MultiClientSC2Manager.create_join_game` to provide a synchronized set of ports for all clients in a multi-client match.
- **Robust Cleanup:** Ensure that `MultiClientEnv` logs critical errors and performs cleanup when a crash occurs during reset or step.

## 3. Acceptance Criteria
- `CaptureTheFlagEnv` can successfully complete the `reset()` and `step()` sequence without crashing the worker process.
- Worker logs are successfully generated in the `logs/` directory for `MultiClientEnv` instances.
- Multi-client communication is established via shared `Portconfig`.
