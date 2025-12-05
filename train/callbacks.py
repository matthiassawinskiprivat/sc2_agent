import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger

class StopTrainingOnMetricThreshold(BaseCallback):
    """
    Stop the training when a monitored metric reaches a threshold.
    :param monitor: The name of the metric to monitor (e.g., 'rollout/ep_rew_mean')
    :param threshold: The threshold value to reach
    :param comparison: The comparison operator ('<', '>', '<=', '>=')
    :param patience: How many consecutive times the condition must be met
    :param verbose:
    """
    def __init__(self, monitor: str, threshold: float, comparison: str = '>', patience: int = 5, verbose: int = 0):
        super(StopTrainingOnMetricThreshold, self).__init__(verbose)
        self.monitor = monitor
        self.threshold = threshold
        self.patience = patience
        self.patience_counter = 0
        self.continue_training = True

        if comparison not in ['<', '>', '<=', '>=']:
            raise ValueError("Invalid comparison operator.")

        self.compare_op = {
            '>': np.greater,
            '<': np.less,
            '>=': np.greater_equal,
            '<=': np.less_equal
        }[comparison]
        self.comparison_str = comparison

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout.
        """
        logger.info("[Callback] _on_rollout_end triggered.")
        # ep_info_buffer provides more up-to-date values for rollout metrics
        if self.monitor == 'rollout/ep_len_mean' and self.model.ep_info_buffer:
            current_value = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
        elif self.monitor == 'rollout/ep_rew_mean' and self.model.ep_info_buffer:
            current_value = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
        elif self.monitor in self.model.logger.name_to_value:
            current_value = self.model.logger.name_to_value[self.monitor]
        else:
            logger.warning(f"[Callback] Metric '{self.monitor}' not found in logger or ep_info_buffer is empty.")
            # Set patience counter to 0 to avoid false positive on next call
            self.patience_counter = 0
            return

        logger.info(f"[Callback DEBUG] Monitored metric '{self.monitor}' value: {current_value:.4f}, Threshold: {self.threshold:.4f}, Comparison: {self.comparison_str}")

        if self.compare_op(current_value, self.threshold):
            self.patience_counter += 1
            logger.info(f"[Callback] Condition met. Patience counter is now {self.patience_counter}/{self.patience}")
        else:
            if self.patience_counter > 0:
                logger.info(f"[Callback] Condition not met. Resetting patience counter from {self.patience_counter} to 0.")
                self.patience_counter = 0
            else:
                logger.info(f"[Callback] Condition not met. Patience counter is already 0.")

        if self.patience_counter >= self.patience:
            logger.success(f"[Callback] Patience reached. Stopping training.")
            self.continue_training = False

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if not self.continue_training:
            logger.error("[Callback] _on_step is returning False. Training should stop.")
        return self.continue_training