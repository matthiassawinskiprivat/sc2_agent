# Specification - Fix Agent Immobility and Action Transmission

## Overview
The SC2 agent is currently immobile during training because movement commands queued by the bot are not being transmitted to the StarCraft II client. This occurred following a refactoring of the training pipeline where the standard `run_game` loop was replaced with a manual simulation control in the environment classes. Additionally, the action masking logic needs verification to ensure efficient training.

## Functional Requirements
- **Action Transmission Fix**: Modify `train/envs/level_0_env.py` (and potentially `minimal_sc2_env.py`, `level_1_env.py`, `level_2_env.py`) to explicitly call `self._client.actions(self._bot._actions)` before calling `self._client.step()`.
- **Bot Action Reset**: Ensure `self._bot._actions.clear()` is called after sending actions to prevent re-sending old commands.
- **SC2Bot Alignment**: Review `src/sc2_bot.py` to ensure it correctly supports the manual action transmission used in the training pipeline.
- **Action Mask Refinement**: Review and refine `UnitControlAgent.get_action_mask` to ensure it accurately restricts the agent to valid movement coordinates, improving training efficiency.

## Non-Functional Requirements
- **TDD Adherence**: All changes must be driven by unit tests.
- **Performance**: Action transmission should be efficient and not introduce significant latency to the training loop.

## Acceptance Criteria
- [ ] The agent moves (randomly or via model) when the training pipeline is executed.
- [ ] Logs or debug statements confirm `client.actions()` is being called with valid commands.
- [ ] Unit tests verify that `Level0Env` correctly sends actions to the mock or real client.
- [ ] Unit tests verify that the action mask correctly filters invalid coordinates.
- [ ] All existing project tests pass.

## Out of Scope
- Major architectural changes to the `burnysc2` library.
- Implementation of new RL algorithms or Master Agent logic.
