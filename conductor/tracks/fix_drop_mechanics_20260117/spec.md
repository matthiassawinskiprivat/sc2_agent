# Specification - Fix Drop Mechanics Masking

## Overview
The drop environment training is currently broken because recent masking changes prevent Medivacs from loading Marines and apply over-aggressive safety margins during unloading. This track fixes the spatial masking logic to restore the "Load -> Fly -> Unload" reward path.

## Problem Statement
1.  **Load Mask Paradox:** The `load` head currently masks everything except allied transports. This prevents a Medivac (the actor) from targeting Marines (the cargo).
2.  **Aggressive Unload Erosion:** The 5x5 erosion kernel in `get_unload_mask` is too restrictive, potentially wiping out valid drop zones on high ground or narrow ledges.
3.  **Incomplete Unload Obstacles:** The `unload` mask only accounts for allied units, ignoring enemy buildings which are the primary drop targets.

## Functional Requirements

### 1. Dynamic Load Masking
The `get_transport_mask` logic must be context-aware:
- If the acting unit **can carry units** (e.g., Medivac), the mask should highlight **loadable allied units** (Marines).
- If the acting unit **is a ground unit**, the mask should highlight **allied transports** (Medivacs).
- Added logic to `SC2Bot` to pass the current unit's type or capabilities to the mask generator.

### 2. Tuned Unload Safety
- Reduce the erosion kernel in `get_unload_mask` from 5x5 to 3x3.
- Ensure the erosion is sufficient to prevent units from being stuck but not so large that it excludes valid landing zones.

### 3. Comprehensive Entity Masking
- Update `SC2Bot.action_masks()` to pass both `self.units` and `self.enemy_structures` (and ideally resources) to `get_unload_mask`.
- Ensure `get_unload_mask` correctly iterates over all provided entities.

## Acceptance Criteria
- [ ] Medivacs can target Marines using the `load` action head.
- [ ] Marines can target Medivacs using the `load` action head.
- [ ] The `unload` mask correctly identifies safe drop zones near enemy buildings without excessive erosion.
- [ ] Integration test `tests/test_drop_mechanics_fix.py` verifies these behaviors.
