# Plan: Agent Performance Optimization Research & Fix

## Phase 1: Deep Investigation & Audit [checkpoint: 7536e17]
*Goal: Identify the primary bottleneck through a detailed analysis of reward, observation, and action logic.*

- [x] Task: Audit Reward Function logic in `level_0_env.py` and base classes for sparsity or magnitude issues.
- [x] Task: Audit Observation Transformer for redundant layers or normalization errors.
- [x] Task: Audit Action Space and Masking implementation in `unit_control_agent.py`.
- [x] Task: Compile "Agent Optimization Report" with a ranked list of potential improvements.
- [ ] Task: Conductor - User Manual Verification 'Deep Investigation & Audit' (Protocol in workflow.md)

## Phase 2: Implementation of Top Improvement
*Goal: Apply the highest-impact fix (Observation Space Pruning) using the TDD workflow.*

- [ ] Task: Implement `LayerPruner` or configuration in `ObservationTransformer` to allow selecting a subset of layers.
- [ ] Task: Update `Level0Env` and `SC2Bot` to use the pruned observation space (e.g., 6 layers instead of 42).
- [ ] Task: Write Failing Tests (Red Phase) for observation pruning and shape consistency.
- [ ] Task: Implement the improvement to pass tests (Green Phase).
- [ ] Task: Refactor implementation and verify code coverage >80%.
- [ ] Task: Run existing integration tests (`tests/test_reward_logic.py`, etc.) to ensure no regressions.
- [ ] Task: Conductor - User Manual Verification 'Implementation of Top Improvement' (Protocol in workflow.md)
