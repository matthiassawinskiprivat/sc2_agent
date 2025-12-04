import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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

        if comparison not in ['<', '>', '<=', '>=']:
            raise ValueError("Invalid comparison operator.")
        
        self.compare_op = {
            '>': np.greater,
            '<': np.less,
            '>=': np.greater_equal,
            '<=': np.less_equal
        }[comparison]
        self.comparison_str = comparison

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Check if the monitored value is in the logs
        if self.monitor in self.model.logger.name_to_value:
            current_value = self.model.logger.name_to_value[self.monitor]
            
            # Check if the condition is met
            if self.compare_op(current_value, self.threshold):
                self.patience_counter += 1
                if self.verbose > 0:
                    print(f"Stopping condition met for '{self.monitor}': {current_value} {self.comparison_str} {self.threshold}. "
                          f"Patience counter: {self.patience_counter}/{self.patience}")
            else:
                self.patience_counter = 0 # Reset counter if condition is not met

            # Check if patience is reached
            if self.patience_counter >= self.patience:
                if self.verbose > 0:
                    print(f"Stopping training because '{self.monitor}' has been {self.comparison_str} {self.threshold} "
                          f"for {self.patience} consecutive checks.")
                return False # Return False to stop training

        return True