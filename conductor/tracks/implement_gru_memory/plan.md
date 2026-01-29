# Plan: GRU Temporal Memory Implementation

- [x] **Research/Analysis**
    - [x] Determine the cleanest way to handle GRU hidden states within the `RNNStates` NamedTuple structure used by `sb3_contrib`.

- [x] **Refactoring Policy**
    - [x] Modify `MaskableRecurrentActorCriticPolicy` in `src/agent/maskable_recurrent_ppo.py`.
    - [x] Update `__init__` to accept `rnn_type`.
    - [x] Update `forward` and `_process_sequence` (if needed) to handle GRU.

- [x] **Refactoring Algorithm & Buffer**
    - [x] Update `MaskableRecurrentPPO._setup_model` to initialize states correctly for GRU.
    - [x] Update `MaskableRecurrentRolloutBuffer` to handle the shape difference.

- [x] **Testing**
    - [x] Add a test case to `tests/test_maskable_recurrent_ppo.py` specifically for `rnn_type="gru"`.
    - [x] Verify that GRU memory persists across steps.
