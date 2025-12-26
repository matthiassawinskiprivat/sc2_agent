# Plan: Agent Performance Optimization Research & Fix

## Phase 1: Deep Investigation & Audit [checkpoint: 7536e17]
*Goal: Identify the primary bottleneck through a detailed analysis of reward, observation, and action logic.*

- [x] Task: Audit Reward Function logic in `level_0_env.py` and base classes for sparsity or magnitude issues.
- [x] Task: Audit Observation Transformer for redundant layers or normalization errors.
- [x] Task: Audit Action Space and Masking implementation in `unit_control_agent.py`.
- [x] Task: Compile "Agent Optimization Report" with a ranked list of potential improvements.
- [ ] Task: Conductor - User Manual Verification 'Deep Investigation & Audit' (Protocol in workflow.md)

## Phase 2: Implementation of Additive Layer Configuration [checkpoint: df44b8b]
*Goal: Implement dynamic, additive observation layers driven by level YAMLs.*

- [x] Task: Create helper logic to load and merge additive layer configurations from YAMLs.
- [x] Task: Update `ObservationTransformer` to support an explicit list of output layers (implemented but needs refinement for this flow).
- [x] Task: Update `Level0Env` to load the config, determine the active layers, and set `observation_space`.
- [x] Task: Update `config/level_0.yaml` (and others) to define the initial layer sets.
- [x] Task: Write Failing Tests (Red Phase) for additive config loading and dynamic environment initialization.
- [x] Task: Implement the logic to pass tests (Green Phase).
- [x] Task: Verify that `Level0Env` uses the correct pruned observation space.
- [x] Task: Conductor - User Manual Verification 'Implementation of Additive Layer Configuration' (Protocol in workflow.md)
