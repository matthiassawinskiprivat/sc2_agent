# Plan: Implement Maskable Recurrent PPO

- [x] **Analysis & Design**
    - [x] Analyze `sb3_contrib.recurrent_ppo.RecurrentPPO` and `sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy` source code (or behavior) to understand how to merge them.
    - [x] Define the class structure for `MaskableRecurrentActorCriticPolicy` and `MaskableRecurrentPPO`.

- [x] **Implementation**
    - [x] Create `src/agent/maskable_recurrent_ppo.py`.
    - [x] Implement `MaskableRecurrentActorCriticPolicy`:
        -   Inherit from `RecurrentActorCriticPolicy`.
        -   Add masking logic to `forward`, `predict`, and `evaluate_actions` (or wherever the distribution is generated).
    - [x] Implement `MaskableRecurrentPPO`:
        -   Inherit from `RecurrentPPO`.
        -   Use `MaskableRecurrentActorCriticPolicy` as the default policy.
        -   Add `action_masks` handling to `predict`.

- [x] **Testing**
    - [x] Create `tests/test_maskable_recurrent_ppo.py`.
    - [x] Test 1: **Masking Validity**. Force a mask where only 1 action is valid and verify the agent picks it 100% of the time.
    - [x] Test 2: **Memory Persistence**. Feed a sequence where the first observation dictates the reward for the last step, but the middle steps are noise. Verify the agent can solve this (simple memory task) with the new class.
    - [x] Test 3: **Integration**. Run a few steps with `LevelDropEnv` (or a mock version) to ensure no crash.

- [x] **Integration**
    - [x] Update `train/train_pipeline.py` to optionally use `MaskableRecurrentPPO` (or just verify it's ready to be swapped in). *Note: The user asked to implement the class, not necessarily replace the production pipeline yet, but we should verify it works.*
