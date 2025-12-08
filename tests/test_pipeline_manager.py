import os
import unittest
from unittest.mock import patch, MagicMock
import yaml

# Add path to import PipelineManager
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))

from train_pipeline import PipelineManager

class TestPipelineManager(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_dir = "tmp_test_models"
        self.models_dir = os.path.join(self.test_dir, "models")
        self.config_dir = os.path.join(self.test_dir, "config")
        os.makedirs(os.path.join(self.models_dir, "level_0"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "level_1"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "level_2"), exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

        self.pipeline_config = {"levels": ["level_0", "level_1", "level_2"]}
        with open(os.path.join(self.config_dir, "pipeline.yaml"), "w") as f:
            yaml.dump(self.pipeline_config, f)

        # Mock level configs
        for i in range(3):
            with open(os.path.join(self.config_dir, f"level_{i}.yaml"), "w") as f:
                yaml.dump({
                    "training": {"num_envs": 1, "training_steps": 100},
                    "env": {"grid_size": 32},
                    "model": {}
                }, f)

    def tearDown(self):
        """Clean up the temporary directory."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_level_status(self):
        """Test the logic for determining level status."""
        manager = PipelineManager(config_dir=self.config_dir, model_dir_base=self.models_dir)

        # Scenario 1: No models in level_0
        status, path = manager.get_level_status("level_0")
        self.assertEqual(status, "NEEDS_TRAINING")
        self.assertIsNone(path)

        # Scenario 2: A checkpoint model exists in level_1
        checkpoint_path = os.path.join(self.models_dir, "level_1", "zerg_1000_steps.zip")
        with open(checkpoint_path, "w") as f:
            f.write("dummy model data")
        
        status, path = manager.get_level_status("level_1")
        self.assertEqual(status, "CONTINUE_TRAINING")
        self.assertEqual(path, checkpoint_path)

        # Scenario 3: A final model exists in level_2
        final_model_path = os.path.join(self.models_dir, "level_2", "final_model.zip")
        with open(final_model_path, "w") as f:
            f.write("dummy final model")

        status, path = manager.get_level_status("level_2")
        self.assertEqual(status, "COMPLETED")
        self.assertEqual(path, final_model_path)
        
        # Scenario 4: Both checkpoint and final model exist in level_1
        final_model_path_l1 = os.path.join(self.models_dir, "level_1", "final_model.zip")
        with open(final_model_path_l1, "w") as f:
            f.write("dummy final model")

        status, path = manager.get_level_status("level_1")
        self.assertEqual(status, "COMPLETED")
        self.assertEqual(path, final_model_path_l1)

    @patch('train_pipeline.PipelineManager._train_level')
    def test_run_pipeline_scenarios(self, mock_train_level):
        """Test the main run method with different pipeline states."""

        # Define a side effect function for the mock to simulate model creation
        def train_side_effect(level_name, model_path_to_load=None):
            model_dir = os.path.join(self.models_dir, level_name)
            final_model_path = os.path.join(model_dir, "final_model.zip")
            with open(final_model_path, "w") as f:
                f.write("dummy trained model")
            return final_model_path
        
        mock_train_level.side_effect = train_side_effect

        # Scenario 1: Clean slate, all levels need training
        manager = PipelineManager(config_dir=self.config_dir, model_dir_base=self.models_dir)
        manager.run()
        self.assertEqual(mock_train_level.call_count, 3)
        mock_train_level.assert_any_call("level_0", model_path_to_load=None)
        
        # After level_0 runs, its final model should be passed to level_1
        final_model_l0 = os.path.join(self.models_dir, "level_0", "final_model.zip")
        mock_train_level.assert_any_call("level_1", model_path_to_load=final_model_l0)

        # After level_1 runs, its final model should be passed to level_2
        final_model_l1 = os.path.join(self.models_dir, "level_1", "final_model.zip")
        mock_train_level.assert_any_call("level_2", model_path_to_load=final_model_l1)

        mock_train_level.reset_mock()
        mock_train_level.side_effect = train_side_effect # re-apply side effect after reset
        # Clean up models for next scenario
        self.tearDown()
        self.setUp()

        # Scenario 2: level_0 is complete, level_1 has a checkpoint
        with open(os.path.join(self.models_dir, "level_0", "final_model.zip"), "w") as f: f.write("final")
        checkpoint_path_l1 = os.path.join(self.models_dir, "level_1", "zerg_5000_steps.zip")
        with open(checkpoint_path_l1, "w") as f: f.write("checkpoint")
        
        manager = PipelineManager(config_dir=self.config_dir, model_dir_base=self.models_dir)
        manager.run()

        # Expected:
        # level_0: Skipped
        # level_1: Continues training from checkpoint
        # level_2: Starts new training, loading from level_1's eventual final model
        self.assertEqual(mock_train_level.call_count, 2)
        mock_train_level.assert_any_call("level_1", model_path_to_load=checkpoint_path_l1)

        # This call assumes level_1 training finishes and saves a final model
        final_model_l1 = os.path.join(self.models_dir, "level_1", "final_model.zip")
        mock_train_level.assert_any_call("level_2", model_path_to_load=final_model_l1)

if __name__ == '__main__':
    unittest.main()
