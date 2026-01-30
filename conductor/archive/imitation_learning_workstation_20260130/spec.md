# Specification: Imitation Learning Workstation (ILW)

## 1. Overview
The Imitation Learning Workstation (ILW) is a comprehensive PyQt5-based tool designed to streamline the "Human-in-the-Loop" training workflow for the StarCraft II agent. It enables users to record gameplay, translate human replays into agent-compatible training data (Observation/Action pairs), verify the integrity of this data via visual playback, and seamlessly integrate it into the Reinforcement Learning pipeline.

## 2. Functional Requirements

### 2.1 UI Application (`app.py`)
- **Framework:** PyQt5.
- **Layout:** Three primary sections/tabs:
    1.  **Record (Play):**
        -   Level Selector (populated from `config/*.yaml`).
        -   Map Selector (filtered by selected level).
        -   "Start Recording" button to launch a Human vs. CPU game.
        -   Auto-save replays to `imitation_learning/replays_raw/<level>/`.
    2.  **Translate:**
        -   List view of Raw Replays (Name, Date, Size).
        -   "Translate" button to process selected replays into `.npz` format.
        -   Progress indication during the computationally expensive replay parsing.
    3.  **Verify:**
        -   List view of Translated Replays (`.npz`).
        -   "Verify (Watch)" button to launch the Validator.
        -   **Visuals:** Must display the Game Client window side-by-side with the `DebugUI` (displaying the *recorded* observation layers) to validate alignment.

### 2.2 Replay Translation Engine (Phase 2)
- **Input:** `.SC2Replay` files.
- **Core Logic:**
    -   Re-simulate the game step-by-step.
    -   Use `SharedObservationTransformer` to generate observations identical to the agent's view.
    -   **Action Mapping:** Convert human actions (clicks) to Agent Grid Actions.
    -   **Decomposition:** Handle multi-unit selection by splitting a single "Group Order" into N distinct training examples (one for each unit in the group).
- **Output:** `.npz` files containing arrays for `observations`, `spatial_actions`, and `categorical_actions`.

### 2.3 Validator (Phase 4)
- **Input:** `.npz` training data.
- **Mechanism:** "Open-Loop to Closed-Loop" verification.
    -   Launch a game with a `ValidatorBot`.
    -   Bot ignores its network and executes the *recorded* actions from the `.npz` file.
    -   **Drift Handling:** The bot attempts to execute the recorded action; if the specific unit is dead or the target invalid due to drift, it logs the discrepancy but continues.
- **Visualization:** Integrates `DebugUI` to render the feature layers stored in the `.npz` file for the current frame, allowing comparison with the live game state.

### 2.4 Training Integration
- **Config Parameters:** Add to `training` section of level configs:
    -   `imitation_data_percentage` (default: 0): Probability (0-100) of sampling a batch from the imitation buffer instead of the online rollout buffer.
    -   `imitation_data_max_steps` (default: `max_steps`): The total number of global training steps during which imitation data is used. After this threshold, `imitation_data_percentage` forces to 0.
- **Logic:**
    -   On startup, check `replays_translated/<level>/`.
    -   If data exists and config allows, load into a dedicated `ImitationReplayBuffer`.
    -   Modify the PPO training loop to sample from this buffer according to the schedule defined above.

## 3. Non-Functional Requirements
- **Performance:** Replay translation is slow; ensuring the UI remains responsive (threading) is critical.
- **Modularity:** The ILW components (Recorder, Translator, Validator) should be decoupled from the main training code where possible, residing in `imitation_learning/`.
- **Consistency:** The `DebugUI` used in verification must match the visual style and layer coloring of the main agent debug tools.

## 4. Acceptance Criteria
- [ ] UI launches and correctly populates Level/Map lists.
- [ ] "Start Recording" saves a playable `.SC2Replay` file.
- [ ] Translation successfully converts a Replay with Group Movement into valid `.npz` data.
- [ ] Validator launches a game, moves units according to the replay, and shows the `DebugUI` with recorded layers.
- [ ] Training loop successfully loads `.npz` data and mixes it into batches (verified via logs showing "Imitation Batch Sampled").
- [ ] `imitation_data_max_steps` correctly stops imitation sampling after the threshold.

## 5. Out of Scope
- **Complex Macro Actions:** We focus on micro (Move, Attack, Load, Unload). Complex build orders or production queuing from replays is out of scope.
- **Perfect Drift Correction:** We accept that validation games will diverge; we do not attempt to "force" the game state to match the replay (rewinding/teleporting units).
