import pytest
import numpy as np
import gymnasium as gym
import torch
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO, MaskableRecurrentActorCriticPolicy

class MockMaskEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.valid_action = 0
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.valid_action = 0
        return np.zeros(10, dtype=np.float32), {}
        
    def step(self, action):
        self.step_count += 1
        # Check if action was valid
        mask = self.action_masks()
        if not mask[action]:
            raise ValueError(f"Agent chose invalid action {action} at step {self.step_count}. Mask: {mask}")
            
        self.valid_action = (self.valid_action + 1) % 5
        return np.zeros(10, dtype=np.float32), 1.0, False, False, {}
        
    def action_masks(self):
        mask = np.zeros(5, dtype=bool)
        mask[self.valid_action] = True
        return mask

def test_maskable_recurrent_ppo_instantiation():
    env = MockMaskEnv()
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=1)
    assert isinstance(model.policy, MaskableRecurrentActorCriticPolicy)

def test_maskable_recurrent_ppo_training():
    """
    Train for a few steps and ensure no invalid actions are taken.
    The MockEnv raises ValueError if an invalid action is taken.
    """
    env = DummyVecEnv([lambda: MockMaskEnv()])
    
    model = MaskableRecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        n_steps=32, 
        batch_size=16, 
        verbose=1,
        policy_kwargs=dict(enable_critic_lstm=False)
    )
    
    # Train for enough steps to trigger updates
    model.learn(total_timesteps=128)

def test_maskable_recurrent_ppo_predict():
    env = MockMaskEnv()
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=1)
    
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    for _ in range(20):
        action_masks = env.action_masks()
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            action_masks=action_masks
        )
        
        # Verify action is valid
        assert action_masks[action], f"Predicted invalid action {action} with mask {action_masks}"
        
        obs, _, _, _, _ = env.step(action)
        episode_starts = np.zeros((1,), dtype=bool)

def test_maskable_recurrent_ppo_memory():
    """
    Test that the hidden state is persistent and affects prediction.
    We create two different sequences and check if the hidden state at the end is different.
    """
    env = MockMaskEnv()
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=0)
    
    obs_1 = np.ones(10, dtype=np.float32) * 0.1
    obs_2 = np.ones(10, dtype=np.float32) * 0.9
    
    # Run two different steps
    _, state_1 = model.predict(obs_1, state=None, episode_start=np.ones(1, dtype=bool))
    _, state_2 = model.predict(obs_2, state=None, episode_start=np.ones(1, dtype=bool))
    
    # Hidden states should be different after different observations
    # state is ((h_pi, c_pi), (h_vf, c_vf)) or ((h_pi,), (h_vf,))
    assert not np.allclose(state_1[0][0], state_2[0][0]), "Hidden states for different observations should be different"
    
    # Now check if state carries over
    _, state_1_after_2 = model.predict(obs_2, state=state_1, episode_start=np.zeros(1, dtype=bool))
    _, state_2_after_2 = model.predict(obs_2, state=state_2, episode_start=np.zeros(1, dtype=bool))
    
    assert not np.allclose(state_1_after_2[0][0], state_2_after_2[0][0]), "State should affect future hidden states"

def test_maskable_recurrent_ppo_gru():
    """
    Test MaskableRecurrentPPO with GRU.
    """
    env = DummyVecEnv([lambda: MockMaskEnv()])
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, rnn_type="gru", verbose=1)
    
    assert model.rnn_type == "gru"
    assert isinstance(model.policy.lstm_actor, torch.nn.GRU)
    
    # Check prediction
    obs = env.reset()
    action, state = model.predict(obs, state=None, episode_start=np.ones(1, dtype=bool))
    assert len(state[0]) == 1 # Only hidden state for GRU
    
    # Check training loop (a few steps)
    model.learn(total_timesteps=64)



if __name__ == "__main__":
    test_maskable_recurrent_ppo_instantiation()
    test_maskable_recurrent_ppo_training()
    test_maskable_recurrent_ppo_predict()
