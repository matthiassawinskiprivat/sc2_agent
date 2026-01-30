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

## Phase 4: Replay Translation Engine (Phase 2 & 3)
- [ ] Task: Implement Replay Simluation Wrapper
    - [ ] Write tests for stepping through a replay using `sc2.Client`
    - [ ] Implement `ReplaySimulator` to extract raw frame data
- [ ] Task: Implement Action Translation Logic
    - [ ] Write tests for coordinate mapping (World -> Grid)
    - [ ] Implement mapping for Move, Attack, Load, Unload heads
    - [ ] Implement Multi-Unit Selection splitting (Decomposition)
- [ ] Task: Implement Observation Reconstruction
    - [ ] Write tests for `SharedObservationTransformer` integration during replay
    - [ ] Implement logic to save `(Obs, Action)` pairs into `.npz` files
- [ ] Task: Conductor - User Manual Verification 'Replay Translation Engine' (Protocol in workflow.md)

## Phase 5: Validator Module
- [ ] Task: Implement ValidatorBot
    - [ ] Write tests for `.npz` data playback logic
    - [ ] Implement `ValidatorBot` that reads from the data queue
- [ ] Task: Integrate DebugUI Visualization
    - [ ] Write tests for feeding recorded layers into `DebugUI`
    - [ ] Implement side-by-side visualization of Game + Debug Layers
- [ ] Task: Conductor - User Manual Verification 'Validator Module' (Protocol in workflow.md)

## Phase 6: Training Integration
- [ ] Task: Update Environment Configuration
    - [ ] Write tests for new `imitation_data_percentage` and `imitation_data_max_steps` fields
    - [ ] Update `train/envs/level_drop_env.py` (and base env) to support these flags
- [ ] Task: Implement Imitation Buffer & Sampling
    - [ ] Write tests for the `ImitationReplayBuffer` (loading `.npz` and batching)
    - [ ] Implement hybrid sampling logic in the PPO training loop
- [ ] Task: Conductor - User Manual Verification 'Training Integration' (Protocol in workflow.md)
