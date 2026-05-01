"""
SelfPlayGlue — asymmetric self-play orchestration for CTF training.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from train.envs.ctf.config import CTFConfig, _SIDE
from train.envs.ctf.env import CTFEnv
from train.envs.ctf.squad_manager import SquadStatus


class SharedCounter:
    """
    Global episode counter with optional distributed all-reduce backend.

    Parameters
    ----------
    backend : {"local", "dist"}
    """

    def __init__(self, backend: str = "local") -> None:
        self._backend = backend
        self._value: int = 0

    @property
    def value(self) -> int:
        return self._value

    def increment(self, n: int = 1) -> int:
        if self._backend == "dist":
            return self._dist_increment(n)
        self._value += n
        return self._value

    def reset(self) -> None:
        self._value = 0

    def _dist_increment(self, n: int) -> int:
        try:
            import torch
            import torch.distributed as dist
            tensor = torch.tensor([n], dtype=torch.long)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            self._value += int(tensor.item())
            return self._value
        except Exception as exc:
            raise RuntimeError(
                "torch.distributed all_reduce failed.  "
                "Ensure the process group is initialised before training."
            ) from exc


@dataclass
class Transition:
    obs: Dict[str, Any]
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    squad_id: str
    episode_id: int


class SelfPlayGlue:
    def __init__(
        self,
        env: CTFEnv,
        learner_policy: Any,
        config: CTFConfig,
        counter_backend: str = "local",
    ) -> None:
        self.env = env
        self.learner_policy = learner_policy
        self.config = config
        self.opponent_policy: Any = self._copy_policy(learner_policy)
        self._global_counter = SharedCounter(backend=counter_backend)
        self._learner_side: _SIDE = "player_a"
        self._episode_id: int = 0

    def run_episode(self) -> Tuple[List[Transition], Dict[str, Any]]:
        transitions: List[Transition] = []
        self._episode_id += 1
        episode_id = self._episode_id

        learner_side = self._learner_side
        opponent_side: _SIDE = "player_b" if learner_side == "player_a" else "player_a"

        obs = self.env.reset(learner_side=learner_side)
        done = False
        total_reward = 0.0
        step = 0

        sm = self.env._squad_manager

        while not done:
            learner_squad = sm.current_learner_squad()
            if learner_squad is None or learner_squad.status != SquadStatus.ACTIVE:
                sm.advance_learner_rr()
                learner_squad = sm.current_learner_squad()

            opp_squad = sm.current_opponent_squad()

            learner_obs = self._obs_for_squad(obs, learner_squad)
            opp_obs = self._obs_for_squad(obs, opp_squad)

            learner_action, log_prob, value = self._learner_forward(learner_obs)

            with self._no_grad_context():
                opp_action, _, _ = self._opponent_forward(opp_obs)

            actions: Dict[str, Any] = {}
            if learner_squad is not None:
                actions[learner_squad.squad_id] = learner_action
            if opp_squad is not None:
                actions[opp_squad.squad_id] = opp_action

            obs, rewards, done, info = self.env.step(actions)

            sm.advance_learner_rr()
            sm.advance_opponent_rr()

            learner_reward = 0.0
            if learner_squad is not None:
                learner_reward = rewards.get(learner_squad.squad_id, 0.0)
            total_reward += learner_reward

            if learner_squad is not None:
                transitions.append(Transition(
                    obs=learner_obs,
                    action=learner_action,
                    log_prob=log_prob,
                    value=value,
                    reward=learner_reward,
                    done=done,
                    squad_id=learner_squad.squad_id,
                    episode_id=episode_id,
                ))

            step += 1

        new_count = self._global_counter.increment(1)
        self._learner_side = "player_b" if learner_side == "player_a" else "player_a"

        if new_count % self.config.every_k_episodes == 0:
            self._sync_opponent_weights()

        episode_info = {
            "episode_id": episode_id,
            "steps": step,
            "learner_side": learner_side,
            "outcome": info.get("outcome"),
            "reason": info.get("reason"),
            "cumulative_learner_reward": total_reward,
            "global_episodes": new_count,
            "opponent_synced": (new_count % self.config.every_k_episodes == 0),
        }

        return transitions, episode_info

    def broadcast_weights(self) -> None:
        if self._global_counter._backend != "dist":
            return
        try:
            import torch
            import torch.distributed as dist
            state_dict = self.learner_policy.get_weights()
            for tensor in state_dict.values():
                dist.broadcast(tensor, src=0)
        except Exception as exc:
            raise RuntimeError("Weight broadcast failed.") from exc

    @property
    def global_episode_count(self) -> int:
        return self._global_counter.value

    def _sync_opponent_weights(self) -> None:
        weights = self.learner_policy.get_weights()
        self.opponent_policy.load_weights(weights)

    def _copy_policy(self, policy: Any) -> Any:
        try:
            return copy.deepcopy(policy)
        except Exception:
            new_policy = policy.__class__.__new__(policy.__class__)
            new_policy.__dict__.update(copy.copy(policy.__dict__))
            new_policy.load_weights(policy.get_weights())
            return new_policy

    def _learner_forward(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
        return self.learner_policy.forward(obs)

    def _opponent_forward(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
        return self.opponent_policy.forward(obs)

    @contextlib.contextmanager
    def _no_grad_context(self):
        try:
            import torch
            with torch.no_grad():
                yield
        except ImportError:
            yield

    def _obs_for_squad(self, obs: Dict[str, Any], squad: Optional[Any]) -> Dict[str, Any]:
        return obs
