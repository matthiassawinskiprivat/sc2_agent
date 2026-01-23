# Plan: Verify Drop Environment Penalties

## Strategy
1.  **Investigate Configuration:**
    *   Check `config/level_drop_0.yaml` (or relevant drop configs) to see if penalties are defined.
    *   Check `src/level_drop_bot.py` or `train/envs/level_drop_env.py` (whichever handles rewards) to see how it reads the config.

2.  **Code Inspection:**
    *   Search for `kill_penalty` and `damage_penalty` in the codebase.
    *   Identify the reward calculation method in the Drop Bot/Env.

3.  **Implementation/Fix:**
    *   If the code uses hardcoded values or ignores the config, update it to use `self.training_conf.get('rewards', {}).get(...)`.
    *   Ensure defaults are reasonable if config is missing.

4.  **Verification:**
    *   Create a script `debug_drop_rewards.py` that loads the config and the bot, then prints the penalty values the bot *thinks* it has.

## Todo List
- [ ] Inspect `config/level_drop_0.yaml` and `config/pipeline_drop.yaml`.
- [ ] Inspect `src/level_drop_bot.py` (or similar) for reward logic.
- [ ] Create `debug_drop_config_check.py` to verify config loading.
- [ ] Apply fixes if values are not propagated.
