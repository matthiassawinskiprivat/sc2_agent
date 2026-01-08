# Specification: Dynamic Unit Costs and Budgeted CTF Spawning

## Overview
Currently, the 'Capture the Flag' (CTF) environment uses a limited unit pool and potentially static/hardcoded unit costs. This track aims to modernize the system by extracting comprehensive unit data (costs, stats) directly from the StarCraft 2 engine using `burnysc2`, updating the `unit_data.json` storage, and implementing a budget-based spawning system in the CTF environment.

## Functional Requirements
1.  **Unit Data Extraction (`generate_unit_json.py`)**:
    *   Update the script to work with the `burnysc2` library.
    *   Extract: Minerals, Vespene, Supply, Build Time, Health, Shields, Weapon Damage, Movement Speed, and Attack Range.
    *   Ensure all units in the `unit_pool` (and common combat units) are captured.
2.  **Configurable Resource Budgets**:
    *   The CTF environment configuration (`.yaml`) must support `mineral_budget` and `vespene_budget` parameters.
3.  **Budgeted Spawning Logic**:
    *   In `CaptureTheFlagSingleEnv`, replace static spawning with a loop that:
        *   Selects a random unit type from the configured `unit_pool`.
        *   Checks if the player has sufficient minerals, vespene, and supply (if applicable) remaining in their budget.
        *   Spawns the unit and deducts its cost if affordable.
        *   Continues until no units in the pool can be afforded or a maximum unit count is reached.
4.  **Configuration Support**:
    *   Ensure `unit_pool: ["ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT"]` (and others) works correctly by mapping string names to game unit types using the extracted data.

## Technical Constraints
*   **Library**: Must use `burnysc2` for game data access.
*   **Storage**: Data must be stored in `data/unit_data.json` for persistence and offline access.
*   **Environment**: Logic must be integrated into `CaptureTheFlagSingleEnv` and its associated bot.

## Acceptance Criteria
*   `generate_unit_json.py` successfully creates/updates `data/unit_data.json` with accurate SC2 stats.
*   CTF environment spawns different army compositions based on the `unit_pool` and `mineral_budget`/`vespene_budget` settings.
*   Changing the budget in the YAML file results in a proportional change in the size/strength of the spawned armies.
*   Units like Reapers and Zealots (previously unused) are correctly spawned and controlled.

## Out of Scope
*   Dynamic unit upgrades (e.g., Stimpack, Combat Shields).
*   Macro-management (building structures or training units during the game).
