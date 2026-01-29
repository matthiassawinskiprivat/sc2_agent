# Level Drop Optimization Track

## Objective
Improve the training efficiency of the Drop Environment (Level 2+) by implementing a "Smart Cutoff" for episodes.

## Problem
Currently, if all transports die, the episode continues until the time limit or total annihilation, even if the remaining ground units are stranded on the starting island. This wastes training steps.

## Solution: Path-Based Stranded Check
We will modify the environment termination logic to end the episode early if:
1.  **All Transports are Dead:** The agent has lost its means of mobility.
2.  **No Ground Path to Objective:** We query the SC2 pathing engine. If the remaining ground units cannot physically walk to the enemy Command Center (because they are stuck on the wrong island), we terminate the episode.

## Implementation Steps
1.  **Update `LevelDropEnv`:** Modify the `_async_step` method in `train/envs/level_drop_env.py`.
2.  **Logic Insertion:** Add a check after the reward calculation but before the final done check.
3.  **Pathing Query:** Use `self._bot.client.query_pathing(start, end)` to verify connectivity.
4.  **Verification:** Run `debug_train_level_7.py` (or create a level 2 specific debug script) to verify the game ends correctly when transports are sacrificed without dropping units.

## Success Criteria
- [ ] Game ends immediately when last transport dies AND units are stranded.
- [ ] Game continues if transport dies AFTER dropping units on the enemy island.
- [ ] No performance regression (pathing queries are fast enough).
