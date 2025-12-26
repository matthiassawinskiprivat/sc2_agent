# Agent Optimization Report

## Ranked Hypotheses for Learning Bottlenecks

### 1. Observation Space Complexity (TOP PRIORITY)
*   **Issue:** The agent currently receives 42 layers of spatial information. For a Level 0 navigation task (move Zergling to Beacon), >90% of these layers are empty (DPS maps, Energy, Cloak, Massive/Armored tags, etc.).
*   **Impact:** The CNN must learn to ignore 38+ irrelevant channels, which massively increases the parameter space and noise during the early stages of training.
*   **Proposed Fix:** Prune the observation space to only essential layers for the current level (e.g., Allied Density, Enemy Density, Pathing, Command Objective).

### 2. Lack of Step Penalty
*   **Issue:** The reward function provides distance-based shaping and a terminal reward, but no negative reinforcement for time taken.
*   **Impact:** The agent may learn to "oscillate" or take inefficient paths because there is no cost to staying in the environment for the full 300 steps.
*   **Proposed Fix:** Add a small negative step reward (e.g., -0.01 or -0.05 per step).

### 3. Reward Signal Magnitude
*   **Issue:** The distance shaping reward `(prev_dist - curr_dist) * 1.0` might be too small per-step compared to the terminal +10 reward, or it might be drowned out if the agent moves slowly.
*   **Impact:** Weak gradient signal for directional movement.
*   **Proposed Fix:** Normalize or scale the distance reward to ensure it provides a clear directional gradient.

### 4. Beacon Visibility & Normalization
*   **Issue:** The Beacon is rendered with a value of 20.0 (before log1p normalization) to make it "stand out".
*   **Impact:** If the network expects 0-1 values, this might cause issues in deeper layers, though `log1p` brings it down to ~3.0.
*   **Proposed Fix:** Ensure consistent normalization across all active layers.

## Selected Improvement for Implementation:
**1. Observation Space Pruning**
We will implement a mechanism to filter/mask the 42 layers down to a minimal set (e.g., 6 core layers) to accelerate CNN convergence.
