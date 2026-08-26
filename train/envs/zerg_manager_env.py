"""Thin Zerg manager env wrapper."""

from train.envs.manager_env import ManagerEnv


class ZergManagerEnv(ManagerEnv):
    """Manager curriculum with Zerg race defaults."""

    def _manager_pre_init(self, training_conf, level_name: str) -> None:
        training_conf.setdefault("race", "zerg")
        super()._manager_pre_init(training_conf, level_name)
