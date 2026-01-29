# Specification: VRAM Caching for PPO Training

## Context
Training `MaskableRecurrentPPO` is currently bottlenecked by the CPU-to-GPU transfer of massive data (Observations and Action Masks) during the gradient update phase. This results in high CPU load (40-50%) and low GPU utilization (10-15%).

## Goal
Implement a VRAM caching mechanism that moves the entire rollout buffer to the GPU once per training iteration, significantly reducing overhead and speeding up the "Double GRU" training.

## Requirements
1.  **VRAM Caching**: Add a mechanism to `MaskableRecurrentRolloutBuffer` to convert all stored numpy arrays into GPU tensors at once.
2.  **Zero-Copy Sampling**: Update `_get_samples` to perform slicing directly on GPU-resident tensors when caching is enabled.
3.  **Automatic Management**: `MaskableRecurrentPPO` should automatically trigger caching at the start of the `train()` phase and invalidate the cache when starting a new `collect_rollouts()` phase.
4.  **Efficiency**: Ensure that no extra copies are made once data is on the GPU.

## Technical Details
-   Target Device: `self.device` (usually "cuda").
-   Data to Cache: `observations`, `actions`, `rewards`, `episode_starts`, `values`, `log_probs`, `advantages`, `returns`, `action_masks`, `mask`.
-   Memory Estimate: For 2048 steps and 65k actions, approximately 1-2 GB of VRAM.
