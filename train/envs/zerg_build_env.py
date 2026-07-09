"""Zerg-specific build environment."""

from train.envs.build_env import BuildEnv


class ZergBuildEnv(BuildEnv):
    """Zerg build curriculum env — thin wrapper setting race defaults."""

    def _build_pre_init(self, training_conf, level_name: str) -> None:
        training_conf.setdefault("race", "zerg")
        training_conf.setdefault("mining_controller", "noop")
        super()._build_pre_init(training_conf, level_name)
