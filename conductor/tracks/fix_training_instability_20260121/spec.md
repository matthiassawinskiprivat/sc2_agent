# Track Spec: Fix Training Instability & Reward Volatility

## Goal
Stabilize the training process by smoothing guidance signals and normalizing event-based rewards to prevent policy collapse and high KL divergence.

## Requirements
1. **Stabilize Meeting Guidance:**
    - Guidance should use a stable reference point (e.g., the **centroid** of the target group) rather than the "closest unit" to prevent penalty spikes when units are loaded or killed.
    - Alternatively, reset the distance tracking immediately when a target disappears.

2. **Normalize Event Rewards:**
    - Use flat rewards for loading and unloading events (e.g., `+1.0` per unit) instead of scaling by `current_cargo_sum`. 
    - This keeps the reward stationary and prevents extreme advantage spikes.

3. **Smooth Phase Transitions:**
    - When a transport transitions from "Meeting" to "Objective" mode, reset its distance tracking to prevent a "teleportation" penalty between the infantry group and the enemy base.

4. **Lower Learning Rate / Adjust KL:**
    - Update `level_drop_0.yaml` to use a more conservative learning rate (e.g., `0.0001`) to handle the remaining volatility.
