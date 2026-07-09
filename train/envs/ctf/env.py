"""
CTFEnv — top-level environment class for the CTF StarCraft II scenario.

Implements the UniversalBaseEnv contract and wires together all CTF
sub-modules: RegionLoader, SpawnController, ObjectiveResolver,
SquadManager, WinConditionChecker, RoleRewardComputer, CTFObsBuilder,
and CTFActionAdapter.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.universal.base_env import (
    ActionSpace,
    BoxActionSpace,
    BoxObservationSpace,
    ObservationSpace,
    UniversalBaseEnv,
)
from train.envs.ctf.config import CTFConfig, _SIDE
from train.envs.ctf.region_loader import RegionLoader
from train.envs.ctf.objective_resolver import ObjectiveResolver
from train.envs.ctf.spawn import SpawnController
from train.envs.ctf.squad_manager import SquadManager, SquadState, Outcome
from train.envs.ctf.win_condition import WinConditionChecker, GameState, TerminalResult
from train.envs.ctf.reward import RoleRewardComputer, StepEvents
from train.envs.ctf.obs_builder import CTFObsBuilder
from train.envs.ctf.action_adapter import CTFActionAdapter


class CTFEnv(UniversalBaseEnv):
    """
    CTF-style StarCraft II training environment (universal pipeline).

    Registered under env_id ``"ctf_sc2_v1"`` (see ``train/envs/ctf/__init__.py``).
    """

    ENV_ID = "ctf_sc2_v1"

    def __init__(self, config: CTFConfig) -> None:
        self.config = config
        self._region_loader = RegionLoader()
        self._reward_computer = RoleRewardComputer(config.rewards)
        self._win_checker = WinConditionChecker()
        self._obs_builder = CTFObsBuilder(config)
        self._action_adapter = CTFActionAdapter(config)

        self._squad_manager: Optional[SquadManager] = None
        self._objective_resolver: Optional[ObjectiveResolver] = None
        self._spawn_controller: Optional[SpawnController] = None
        self._learner_side: _SIDE = "player_a"
        self._step_count: int = 0
        self._episode_rng: Optional[np.random.Generator] = None

        self._sc2_client: Optional[Any] = None
        self._game_thread: Optional[threading.Thread] = None
        self._connected = False

        self._action_space: Optional[ActionSpace] = None
        self._observation_space: Optional[ObservationSpace] = None

    def reset(self, *, learner_side: Optional[_SIDE] = None, **kwargs: Any) -> Dict[str, Any]:
        if learner_side is not None:
            self._learner_side = learner_side

        seed = self.config.episode_seed
        self._episode_rng = np.random.default_rng(seed)
        self._step_count = 0

        regions = self._region_loader.load(self.config.map_path)

        self._squad_manager = SquadManager(
            squad_configs=self.config.squads,
            learner_side=self._learner_side,
            regions=regions,
        )

        self._spawn_controller = SpawnController(
            config=self.config,
            regions=regions,
            rng=self._episode_rng,
        )

        if not self._connected:
            self._connect()

        self._reset_game()

        self._objective_resolver = ObjectiveResolver(
            config=self.config,
            regions=regions,
            sc2_client=self._sc2_client,
        )
        self._objective_resolver.resolve()

        self._squad_manager.bind_objectives(self._objective_resolver)

        return self._build_obs()

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        assert self._squad_manager is not None, "Call reset() before step()."

        self._step_count += 1
        sm = self._squad_manager

        active_squads = sm.active_learner_squads() + sm.active_opponent_squads()
        sc2_commands = self._action_adapter.to_sc2_commands(
            actions=actions,
            squads=active_squads,
            objective_resolver=self._objective_resolver,
        )
        raw_events = self._execute_commands(sc2_commands)

        sm.process_events(raw_events)
        sm.check_and_reassign(self._region_loader.get_cached(self.config.map_path))

        gs = self._build_game_state()

        events_per_squad = self._extract_learner_events(raw_events, sm)
        rewards: Dict[str, float] = {}
        for sq_state in sm.learner_squad_states():
            events = events_per_squad.get(sq_state.squad_id, StepEvents())
            r = self._reward_computer.compute(sq_state, events, gs)
            rewards[sq_state.squad_id] = r

        terminal: TerminalResult = self._win_checker.check(gs)
        obs = self._build_obs()

        info: Dict[str, Any] = {
            "obs_version": self.get_obs_version(),
            "step": self._step_count,
            "outcome": terminal.outcome if terminal.done else None,
            "reason": terminal.reason if terminal.done else None,
            "learner_side": self._learner_side,
        }

        return obs, rewards, terminal.done, info

    def close(self) -> None:
        self._connected = False
        if self._sc2_client is not None:
            try:
                self._sc2_client.close()
            except Exception:
                pass
            self._sc2_client = None
        if self._game_thread is not None and self._game_thread.is_alive():
            self._game_thread.join(timeout=5.0)
            self._game_thread = None

    def seed(self, seed: Optional[int] = None) -> None:
        self._episode_rng = np.random.default_rng(seed)

    def get_obs_version(self) -> str:
        return "ctf_v1"

    @property
    def action_space(self) -> ActionSpace:
        if self._action_space is None:
            n = self.config.max_units_per_squad
            low = np.array([-1.0, -1.0, 0.0, 0.0] * n, dtype=np.float32).reshape(n, 4)
            high = np.array([1.0, 1.0, float(self.config.max_enemies_visible - 1), 2.0] * n,
                            dtype=np.float32).reshape(n, 4)
            self._action_space = BoxActionSpace(low, high)
        return self._action_space

    @property
    def observation_space(self) -> ObservationSpace:
        if self._observation_space is None:
            H = self.config.global_grid_size
            W = self.config.global_grid_size
            shape = (10 * H * W + 7 + self.config.max_units_per_squad,)
            self._observation_space = BoxObservationSpace(shape, low=0.0, high=1.0)
        return self._observation_space

    def _connect(self) -> None:
        self._connected = True

    def _reset_game(self) -> None:
        if self._spawn_controller is not None:
            self._spawn_controller.generate_rosters(self._squad_manager)

    def _execute_commands(self, commands: List[Any]) -> List[Any]:
        if self._sc2_client is None:
            return []
        return []

    def _extract_learner_events(
        self, raw_events: List[Any], sm: SquadManager
    ) -> Dict[str, "StepEvents"]:
        result: Dict[str, StepEvents] = {}
        for ev in raw_events:
            squad_id = sm.unit_to_squad(getattr(ev, "attacker_tag", None))
            if squad_id and sm.is_learner_squad(squad_id):
                se = result.setdefault(squad_id, StepEvents())
                se.absorb(ev)
        return result

    def _build_game_state(self) -> GameState:
        sm = self._squad_manager
        return GameState(
            step=self._step_count,
            max_steps=self.config.max_steps,
            learner_pylons=sm.learner_pylon_count(),
            learner_units_alive=sm.learner_unit_count(),
            enemy_pylons_alive=sm.opponent_pylon_count(),
            enemy_units_alive=sm.opponent_unit_count(),
            learner_squads=sm.learner_squad_states(),
        )

    def _build_obs(self) -> Dict[str, Any]:
        sm = self._squad_manager
        active = sm.current_learner_squad()
        return self._obs_builder.build(
            active_squad=active,
            squad_manager=sm,
            step=self._step_count,
        )
