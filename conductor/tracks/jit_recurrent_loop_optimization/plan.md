# Plan: JIT Recurrent Loop Implementation

- [ ] **Implementation**
    - [ ] Create JIT-scriptable functions in `src/agent/maskable_recurrent_ppo.py`:
        -   `_lstm_process_sequence_jit`: Optimized loop for LSTM.
        -   `_gru_process_sequence_jit`: Optimized loop for GRU.
    - [ ] Modify `MaskableRecurrentActorCriticPolicy._process_sequence` to delegate to these JIT functions.

- [ ] **Verification**
    - [ ] Run `tests/test_maskable_recurrent_ppo.py` to ensure logic remains identical (no regression).
    - [ ] (Manual) Verify Training FPS increase.

- [ ] **Polishing**
    - [ ] Clean up any debug prints or redundant logic.
