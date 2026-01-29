# Specification: Advanced Drop Reward Shaping

## Context
The agent has successfully learned to load units but has plateaued. It tends to load a single unit, fly to the drop zone, and unload it, often leading to the unit's death without significant impact. This behavior is suboptimal compared to a "full drop" strategy.

## Goals
1.  **Incentivize Full Loads:** Reward the agent progressively more for each additional unit loaded into the transport.
2.  **Incentivize Movement (Shaping):** Provide a dense reward for moving towards the drop zone, but *only* when the transport has cargo.
3.  **Cargo-Weighted Travel:** Scale the movement reward by the amount of cargo, teaching the agent that "time is money" specifically when carrying a full load.

## Requirements
1.  **Modify `LevelDropBot.calculate_drop_reward`**:
    -   **Progressive Loading Reward**: Instead of a flat `load_reward`, use a multiplier based on current cargo. E.g., `reward = base_load_reward * (current_cargo + 1)`.
    -   **Distance Reward**: Calculate the change in distance to the objective (`old_dist - new_dist`).
    -   **Cargo Weighting**: Multiply the distance reward by `cargo_used`. `dist_reward = delta_dist * 0.01 * cargo_used`.
2.  **Configuration**: Add necessary coefficients to `level_drop_0.yaml` (or use sensible defaults in the bot).

## Technical Details
-   The reward logic is in `src/level_drop_bot.py`.
-   We need to track `previous_distance` to the objective for the transport unit specifically. Since there might be multiple Medivacs (though usually 1 in Level Drop 0), we should sum or average this, or focus on the "best" Medivac. For simplicity in Level 0 (1 Medivac), we can track the Medivac's distance.
