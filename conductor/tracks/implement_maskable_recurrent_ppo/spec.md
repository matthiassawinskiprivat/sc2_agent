# Specification: Maskable Recurrent PPO

## Context
The user wants to combine the memory capabilities of `RecurrentPPO` (from `sb3-contrib`'s `RecurrentPPO` or `stable_baselines3`'s `PPO` with LSTM support if available, but `sb3-contrib` is the standard for RecurrentPPO) with the valid action masking of `MaskablePPO` (from `sb3-contrib`).
Currently, `sb3-contrib` provides these as separate algorithms. To enable an agent that can both "remember" (using LSTM/GRU) and "respect rules" (using action masks), we need a hybrid implementation.

## Goal
Implement a `MaskableRecurrentPPO` class that inherits from `RecurrentPPO` but integrates the action masking logic during the sampling and training phases, similar to `MaskablePPO`.

## Requirements
1.  **Class Definition**: Create `MaskableRecurrentPPO` in `src/agent/maskable_recurrent_ppo.py`.
2.  **Inheritance**: It should inherit from `RecurrentPPO` (from `sb3_contrib`).
3.  **Masking Support**:
    -   Override `predict` to accept an `action_masks` argument.
    -   Override the policy's `forward` or sampling method to apply masks to the logits before sampling.
    -   Ensure compatibility with `ActionMasker` wrapper or the environment's `action_masks()` method.
4.  **Testing**:
    -   Unit tests to verify that invalid actions are never chosen when masked.
    -   Unit tests to verify that the hidden state is maintained and affects decisions (memory check).
    -   Integration test ensuring it can run in the existing pipeline (at least instantiation and one step).

## Technical Details
-   **Base Class**: `sb3_contrib.ppo_recurrent.RecurrentPPO`
-   **Policy**: We might need a `MaskableRecurrentActorCriticPolicy`.
-   **Masking Logic**: The masking usually happens in the `distribution` method or by modifying the logits before creating the distribution. In `MaskablePPO`, it uses `MaskableActorCriticPolicy`. We need to fuse `RecurrentActorCriticPolicy` with `MaskableActorCriticPolicy`.

## Constraints
-   No changes to existing `MaskablePPO` or `RecurrentPPO` library code.
-   Must work with `gymnasium` spaces.
-   Must be fully tested.
