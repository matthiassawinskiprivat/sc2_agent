from train.envs.level_0_env import Level0Env
from train.envs.level_3_env import Level3Env
from train.envs.level_7_env import Level7Env

# Universal-pipeline CTF environment lives in train.envs.ctf.
# It registers itself in src.universal.env_registry.EnvRegistry when imported:
#   import train.envs.ctf        # registers "ctf_sc2_v1"
# It is intentionally NOT wired into train_pipeline.py's ENV_REGISTRY because
# it uses CTFConfig (pydantic) rather than the training_conf dict interface.
