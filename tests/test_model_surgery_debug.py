import yaml
import os
import torch
import gymnasium as gym
from sb3_contrib import MaskablePPO
from src.agent.models.unit_control_network import UnitControlNetwork

def test_model_shape():
    with open("config/level_drop_0.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_params = config["model"]
    training_params = config["training"]
    env_params = config["env"]
    
    pk = model_params["policy_kwargs"]
    pk["net_arch"] = []
    if pk.get("features_extractor_class") == "src.agent.models.unit_control_network.UnitControlNetwork":
        pk["features_extractor_class"] = UnitControlNetwork
        
    import numpy as np
    obs_space = gym.spaces.Box(0, 255, (35, 128, 128), dtype=np.uint8)
    action_space = gym.spaces.Discrete(400)
    
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = obs_space
            self.action_space = action_space
        def reset(self, seed=None): return obs_space.sample(), {}
        def step(self, action): return obs_space.sample(), 0, False, False, {}

    env = DummyEnv()
    
    model = MaskablePPO(
        policy=model_params["policy"],
        env=env,
        policy_kwargs=pk,
        device="cpu"
    )
    
    print(f"Policy type: {type(model.policy)}")
    print(f"Value net shape: {model.policy.state_dict()['value_net.weight'].shape}")
    print(f"Features dim: {model.policy.features_extractor.features_dim}")
    print(f"Net arch: {model.policy.net_arch}")

if __name__ == "__main__":
    test_model_shape()
