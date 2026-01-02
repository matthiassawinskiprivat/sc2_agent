# Specification: Fix Level 7/8 Spawning Reliability and Unit Ratios

## Overview
This track addresses two primary issues in the spawning logic of the Level 7 (and derived Level 8) combat environments:
1. **Incorrect Unit Ratios:** Zerglings are currently spawned 1-to-1 with allied units, whereas they should be 2-to-1 to match StarCraft II balance.
2. **Spawning Failures & Accumulation:** Units occasionally fail to spawn or accumulate from previous episodes, caused by Fog of War hiding units during cleanup and race conditions in the reset sequence.

## Functional Requirements

### 1. Centralized Robust Reset
- Refactor the reset logic into the `SC2Bot` base class to ensure all bots benefit from reliable episode initialization.
- **Vision Guarantee:** Call `debug_show_map()` followed immediately by `self.client.step()` and a state refresh at the very start of the reset process.
- **Robust Cleanup:** Implement a verification loop that kills all units and steps the game until `self.all_units` is empty.

### 2. Spawning with Verification
- **Ratio Logic:** Implement unit-specific ratios (e.g., 2-to-1 for Zerglings) in the spawning loop.
- **Post-Spawn Verification:** After spawning commands and a game step, verify that the actual counts match the expected counts.
- **Retry Mechanism:** If counts don't match, trigger a full cleanup and retry the spawning process (up to 3 attempts).

### 3. Location Safety
- Use `random.sample` with sufficient padding to prevent placement collisions.

## Non-Functional Requirements
- **Robustness:** 100% reliable unit counts and map vision across all episodes.
- **Performance:** Minimal overhead by only stepping when state changes are required.

## Acceptance Criteria
- `SC2Bot` base class handles cleanup and vision initialization reliably.
- `Level7Bot` correctly spawns 2 Zerglings for every 1 Allied unit when the enemy type is `ZERGLING`.
- No units persist between episodes.
- Fog of War is consistently disabled at the start of every episode.

## Out of Scope
- Modifying observation layers.
- Changing game difficulty.