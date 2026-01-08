# Track: Drop Operations Prototype

Implement a vertical slice to prove the agent can learn complex hierarchical and mechanical tasks (Load -> Move -> Unload) using the new Reasoning Core and Dynamic Action Heads.

**Core Philosophy:** Test-Driven Development (TDD). Every component must have a unit/integration test before full integration.

## Goals
- [x] Add `DROP` command to the protocol.
- [x] Enhance Observation Pipeline with transport-specific layers.
- [x] Upgrade `UnitControlAgent` with a `ReasoningCore` (3-step GRU loop).
- [x] Implement Dynamic Action Heads (Move, Attack, Load, Unload).
- [x] Create `Level_Drop_Env` using specific user-provided maps.
- [x] Implement Weight Surgery to preserve Level 7 skills, while supporting fresh training from Level 0.

## Detailed Implementation Plan

### 1. Protocol & Constants (TDD: `tests/test_protocol.py`)
- [x] **Update `src/agent/commands.py`**
    - Add `CommandType.DROP` to the Enum.
- [x] **Update `src/utils/layer_constants.py`**
    - Add `transport_slots` (Index ~42).
    - Add `passenger_density` (Index ~43).
- [x] **Test:** Verify Enum values and Layer indices do not overlap.

### 2. Observation Pipeline (TDD: `tests/test_observation_transformer.py`)
- [x] **Update `src/transformers/observation_transformer.py`**
    - Implement `_draw_transport_info(self, raw_obs, layers)`.
    - **Logic:**
        - `transport_slots`: 0.5 (Empty) / 1.0 (Full) for Medivac/Overlord/WarpPrism.
        - `passenger_density`: Draw integer count of passengers at transport location.
- [x] **Test:** Create a mock observation with 1 Full Medivac and 1 Empty Medivac. Assert layers 42 and 43 contain expected values at correct pixel coordinates.

### 3. Dynamic Network Architecture (TDD: `tests/test_dynamic_action_head.py`)
- [x] **Create `src/models/components/dynamic_action_head.py`**
    - `DynamicActionHead(nn.Module)`:
        - Accepts `head_names: List[str]`.
        - Dynamically creates `nn.Linear` layers for each name in `head_names`.
- [x] **Refactor `src/agent/unit_control_agent.py`** (The wrapper) and **Model Definition** (in `train/` or `src/models`)
    - Ensure `action_heads` are read from config.
    - **Accumulation Logic:** Ensure `action_heads` are cumulative across levels (handled in Config Manager or Pipeline).
        - Level 0: `['move']`
        - Level 7: `['attack']` -> Result: `['move', 'attack']`
        - Drop Env: `['load', 'unload']` -> Result: `['move', 'attack', 'load', 'unload']`
- [x] **Test:** Initialize model with different config lists. Check `model.policy.action_net` output shapes match expected heads.

### 4. Weight Surgery & Initialization (TDD: `tests/test_model_surgery.py`)
- [x] **Update `src/utils/model_utils.py`**
    - Implement `load_flexible_checkpoint(model, checkpoint_path)`.
    - **Case A (New Training / Level 0):** If `checkpoint_path` is None, initialize all heads (Move, Attack, Load, Unload) randomly (standard PyTorch init).
    - **Case B (Surgery / Transfer):** If `checkpoint_path` exists:
        - Copy CNN & Core weights.
        - Match heads by name: Copy `move` weights to `move` head.
        - Initialize mismatched heads (`load`, `unload`) randomly.
        - **Selector Head:** Copy weights for indices 0,1 (Move/Attack), init 2,3 (Load/Unload) randomly.
- [x] **Test:** Load a "Move-Only" checkpoint into a "Move+Attack" model. Assert Move weights match, Attack weights are random.

### 5. Environment Implementation (TDD: `tests/test_level_drop_env.py`)
- [x] **Create `src/envs/level_drop_env.py`**
    - **Map:** Use user-provided map `DropOperations.SC2Map`.
    - **Setup:**
        - Identify "Landing Zone": Find `UnitTypeId.COMMANDCENTER`.
        - Identify "Squad": Find Marines/Medivacs.
    - **Visualization:**
        - Draw Command Center on `COMMAND_OBJECTIVE` (Layer 41).
        - Draw Squad units on `COMMAND_ASSIGNMENT` (Layer 40).
    - **Reward Function:**
        - `+ Reward`: Unit entering Transport (Loading).
        - `+ Reward`: Unit exiting Transport (Unloading).
        - `+ Reward`: Damage dealt to enemy Units/Structures.
        - `+ Reward`: Enemy Units/Structures killed.
- [x] **Test:** Mock SC2 GameState. Call `step()`. Assert rewards trigger correctly for Load/Unload events.

### 6. Configuration & Verification
- [x] **Create `config/level_drop.yaml`**
    - `active_layers`: `['transport_slots', 'passenger_density']` (Added to existing).
    - `action_heads`: `['load', 'unload']` (Added to existing `move`, `attack`).
- [x] **Verify Accumulation:** Ensure `ConfigManager` correctly merges this into `['move', 'attack', 'load', 'unload']`.