# Implementation Plan: Imitation Learning Workstation (ILW)

## Phase 1: Infrastructure & Directory Setup [checkpoint: ffb09e2]
- [x] Task: Create directory structure for ILW 48184c0
    - [x] Create `imitation_learning/`
    - [x] Create `imitation_learning/replays_raw/`
    - [x] Create `imitation_learning/replays_translated/`
- [x] Task: Implement Config Resolver 316e512
    - [x] Write tests for `imitation_learning/config_resolver.py` (ensure it parses `level_*.yaml` correctly for action heads and layers)
    - [x] Implement `ConfigResolver` class
- [x] Task: Conductor - User Manual Verification 'Infrastructure & Directory Setup' (Protocol in workflow.md)

## Phase 2: PyQt5 UI Skeleton [checkpoint: b556e81]
- [x] Task: Implement Main Window Layout 5e2f55b
    - [x] Write tests for UI component initialization
    - [x] Create `imitation_learning/app.py` with 3 tabs: Record, Translate, Verify
- [x] Task: Implement Level and Map Selectors 5e2f55b
    - [x] Write tests for dynamic map filtering based on level selection
    - [x] Implement reactive dropdowns for Level and Map selection
- [x] Task: Conductor - User Manual Verification 'PyQt5 UI Skeleton' (Protocol in workflow.md)

## Phase 3: Recorder Module
- [x] Task: Implement Game Launcher a7fe24b
    - [x] Write tests for `sc2.run_game` parameter generation
    - [x] Implement "Start Recording" logic that launches Human vs. CPU
- [x] Task: Implement Post-Game File Management 268f228
    - [x] Write tests for automatic file movement and renaming (with timestamps)
    - [x] Implement logic to move `.SC2Replay` files to `replays_raw/<level>/`
- [x] Task: Conductor - User Manual Verification 'Recorder Module' (Protocol in workflow.md)

## Phase 4: Replay Translation Engine (Phase 2 & 3) [checkpoint: 1be37ac]
- [x] Task: Implement Replay Simluation Wrapper ffd8ea1
    - [x] Write tests for stepping through a replay using `sc2.Client`
    - [x] Implement `ReplaySimulator` to extract raw frame data
- [x] Task: Implement Action Translation Logic ffd8ea1
    - [x] Write tests for coordinate mapping (World -> Grid)
    - [x] Implement mapping for Move, Attack, Load, Unload heads
    - [x] Implement Multi-Unit Selection splitting (Decomposition)
- [x] Task: Implement Observation Reconstruction ffd8ea1
    - [x] Write tests for `SharedObservationTransformer` integration during replay
    - [x] Implement logic to save `(Obs, Action)` pairs into `.npz` files
- [x] Task: Conductor - User Manual Verification 'Replay Translation Engine' (Protocol in workflow.md)

## Phase 5: Validator Module
- [x] Task: Integrate DebugUI Visualization 8ef033d
    - [x] Write tests for feeding recorded layers into `DebugUI`
    - [x] Implement side-by-side visualization of Game + Debug Layers
- [x] Task: Conductor - User Manual Verification 'Validator Module' (Protocol in workflow.md)

## Phase 6: Training Integration [checkpoint: 8ef033d]
- [x] Task: Update Environment Configuration 8ef033d
    - [x] Write tests for new `imitation_data_percentage` and `imitation_data_max_steps` fields
    - [x] Update `train/envs/level_drop_env.py` (and base env) to support these flags
- [x] Task: Implement Imitation Buffer & Sampling 8ef033d
    - [x] Write tests for the `ImitationReplayBuffer` (loading `.npz` and batching)
    - [x] Implement hybrid sampling logic in the PPO training loop
- [x] Task: Conductor - User Manual Verification 'Training Integration' (Protocol in workflow.md)
