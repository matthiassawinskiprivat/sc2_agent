
import warnings
from typing import Any, Dict, Optional, Type, Union

import torch
import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO as SB3MaskablePPO
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance

from imitation_learning.buffer import ImitationReplayBuffer

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import (
    MaskableDistribution,
    MaskableMultiCategoricalDistribution,
    MaskableBernoulliDistribution,
    MaskableCategoricalDistribution,
)
import gymnasium as gym

class CustomMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom MaskableActorCriticPolicy that ensures compatibility with MaskablePPO.
    """
    def __init__(self, *args, **kwargs):
        # Remove recurrent-specific arguments if they leaked into non-recurrent policy
        kwargs.pop("lstm_hidden_size", None)
        kwargs.pop("n_lstm_layers", None)
        kwargs.pop("shared_lstm", None)
        kwargs.pop("enable_critic_lstm", None)
        kwargs.pop("rnn_type", None)
        super().__init__(*args, **kwargs)

class MaskablePPO(SB3MaskablePPO):
    """
    Custom MaskablePPO implementation that supports:
    1. Imitation Learning from expert demonstrations
    """
    policy_class = CustomMaskableActorCriticPolicy

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        # New parameters for imitation learning
        imitation_data_percentage: float = 0.0,
        imitation_data_max_steps: int = 0,
        imitation_coef: float = 0.5,
        level_name: Optional[str] = None,
        **kwargs,
    ):
        # FORCE verbose to be an int. If it's a string, we MUST convert it before SB3 sees it.
        try:
            if isinstance(verbose, str):
                verbose = int(verbose)
            else:
                verbose = int(verbose)
        except:
            verbose = 1
            
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        self.imitation_data_percentage = imitation_data_percentage
        self.imitation_data_max_steps = imitation_data_max_steps
        self.imitation_coef = imitation_coef
        self.level_name = level_name
        self.imitation_buffer = None
        
        if _init_setup_model:
            self._setup_imitation()

    def _setup_model(self) -> None:
        # SB3's MaskablePPO._setup_model checks if self.policy is an instance of MaskableActorCriticPolicy.
        # If it's not set yet, it will create it from self.policy_class.
        
        # We ensure the policy class is set correctly
        if self.policy_class is None or not issubclass(self.policy_class, MaskableActorCriticPolicy):
            self.policy_class = CustomMaskableActorCriticPolicy
            
        super()._setup_model()
        
        # Ensure the action distribution supports masking
        if isinstance(self.action_space, gym.spaces.Discrete):
             self.policy.action_dist = MaskableCategoricalDistribution(self.action_space.n)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
             self.policy.action_dist = MaskableMultiCategoricalDistribution(self.action_space.nvec)
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
             self.policy.action_dist = MaskableBernoulliDistribution(self.action_space.n)

    def _setup_imitation(self, level_name: Optional[str] = None, 
                         imitation_data_percentage: Optional[float] = None,
                         imitation_data_max_steps: Optional[int] = None,
                         imitation_coef: Optional[float] = None):
        """Initializes the imitation buffer if data exists for the current level."""
        
        # Apply Overrides
        if imitation_data_percentage is not None: self.imitation_data_percentage = imitation_data_percentage
        if imitation_data_max_steps is not None: self.imitation_data_max_steps = imitation_data_max_steps
        if imitation_coef is not None: self.imitation_coef = imitation_coef

        if self.imitation_data_percentage <= 0:
            self.imitation_buffer = None
            return

        # Priority: 1. Argument, 2. Instance Attribute
        level_name = level_name or getattr(self, "level_name", None)
        
        # Fallback to env resolution if not provided
        if not level_name and self.env is not None:
            # VecEnv might have 'level_name' attribute if it's SyncVecEnv/SubprocVecEnv wrapping our envs
            level_name = getattr(self.env, "level_name", None)
            
            # Try to get from first wrapped env if it's a VecEnv
            if not level_name and hasattr(self.env, "get_attr"):
                try:
                    attr_vals = self.env.get_attr("level_name", indices=[0])
                    if attr_vals:
                        level_name = attr_vals[0]
                except:
                    pass
            
        if level_name:
            self.imitation_buffer = ImitationReplayBuffer(level_name)
            if self.imitation_buffer.total_size == 0:
                self.imitation_buffer = None
            else:
                logger.info(f"Imitation learning ENABLED for level {level_name} (Percentage: {self.imitation_data_percentage}%, Max Steps: {self.imitation_data_max_steps})")
        else:
            self.imitation_buffer = None

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Overridden to include imitation loss.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        imitation_losses = []

        continue_training = True

        # train for n_epochs episodes
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                    actions = rollout_data.actions.long()

                # Distribution-specific processing if needed (MaskablePPO handles this in evaluate_actions)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, 
                    actions, 
                    action_masks=rollout_data.action_masks if hasattr(rollout_data, "action_masks") else None
                )
                
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2))

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Value clipping
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the surrogate L_VF
                value_loss = torch.mean((rollout_data.returns - values_pred) ** 2)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # --- IMITATION LOSS ---
                imitation_loss = torch.tensor(0.0, device=self.device)
                if self.imitation_buffer and self.num_timesteps < self.imitation_data_max_steps:
                    if np.random.rand() < (self.imitation_data_percentage / 100.0):
                        expert_data = self.imitation_buffer.sample_batch(self.batch_size)
                        if expert_data:
                            expert_obs, expert_acts, expert_masks = expert_data
                            
                            if self._n_updates % 10 == 0:
                                logger.info(f"Imitation Sampling: Batch {self.batch_size}, Step {self.num_timesteps}/{self.imitation_data_max_steps}")
                                
                            expert_obs_tensor = torch.as_tensor(expert_obs, device=self.device).float()
                        expert_acts_tensor = torch.as_tensor(expert_acts, device=self.device).long()
                        
                        expert_masks_tensor = None
                        if expert_masks is not None:
                            expert_masks_tensor = torch.as_tensor(expert_masks, device=self.device).bool()
                            # BAND-AID: "Trust the Expert"
                            # Ensure the action the expert actually took is marked as VALID in the mask.
                            # This prevents infinite loss if the heuristic mask and expert action disagree.
                            if len(expert_acts_tensor.shape) == 1:
                                # Discrete action space
                                expert_masks_tensor.scatter_(1, expert_acts_tensor.unsqueeze(1), True)
                            else:
                                # MultiDiscrete - would need more complex logic, but let's assume Discrete for now
                                pass

                        # Evaluate expert actions using current policy
                        _, expert_log_prob, _ = self.policy.evaluate_actions(
                            expert_obs_tensor, 
                            expert_acts_tensor,
                            action_masks=expert_masks_tensor
                        )
                        
                        # Supervised loss: minimize -log_prob of expert actions
                        imitation_loss = -torch.mean(expert_log_prob) * self.imitation_coef
                        imitation_losses.append(imitation_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + imitation_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {self.num_timesteps} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if imitation_losses:
            self.logger.record("train/imitation_loss", np.mean(imitation_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    @classmethod
    def load(cls, path: str, env: Optional[GymEnv] = None, device: Union[torch.device, str] = "auto", custom_objects: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Overridden load to automatically handle architecture mismatches (e.g. Recurrent -> Non-Recurrent).
        This makes the class a drop-in replacement that fixes loading errors without pipeline changes.
        """
        try:
            # Try standard load first
            return super().load(path, env=env, device=device, custom_objects=custom_objects, **kwargs)
        except (RuntimeError, ValueError, TypeError) as e:
            # If it fails, retry with non-strict parameter loading and no optimizer
            logger.warning(f"Standard load failed, attempting architecture-agnostic recovery: {e}")
            
            if custom_objects is None:
                custom_objects = {}
            
            # Signal SB3 to skip optimizer loading
            custom_objects["optimizer"] = None
            
            # Temporary monkeypatch to force non-strict loading
            original_set_params = cls.set_parameters
            def non_strict_set_params(self, params, exact_match=True, device=None):
                return original_set_params(self, params, exact_match=False, device=device)
            
            try:
                cls.set_parameters = non_strict_set_params
                return super().load(path, env=env, device=device, custom_objects=custom_objects, **kwargs)
            finally:
                cls.set_parameters = original_set_params
