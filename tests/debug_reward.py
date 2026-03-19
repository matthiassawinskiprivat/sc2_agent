
from train.envs.level_3_env import Level3Env
from unittest.mock import MagicMock
import asyncio

async def test_env():
    env = Level3Env(level_name="level_move_multi_0", num_workers=1, training_conf={"config_dir": "config_universal", "rewards": {"success": 40.0, "step_penalty": -0.1, "distance_multiplier": 1.0}})
    
    # Mock bot methods
    class MockBot:
        def __init__(self):
            self.game_info = MagicMock()
            self.pending_success_rewards = {}
            self.current_assignments = {1: 3, 2: 4}
            self.completed_tags = set()
            self._actions = []
            
        def get_active_zerglings(self):
            u1 = MagicMock(); u1.tag = 1
            u2 = MagicMock(); u2.tag = 2
            return [u1, u2]
            
        async def apply_action(self, tag, action):
            pass
            
        def _prepare_step(self, state, proto):
            pass
            
        def calculate_distance_reward(self, tag):
            return 1.5
            
        async def check_collections(self):
            # Simulate unit 1 reaching beacon 3
            self.pending_success_rewards[1] = 20.0
            
        def get_observation_for_unit(self, tag):
            return {"map": 0}, {}

    env._bot = MockBot()
    env.active_unit_tags = [1, 2]
    env.current_unit_index = 0
    env._client = MagicMock()
    
    async def mock_step(): pass
    env._client.step = mock_step
    
    async def mock_obs():
        obs = MagicMock()
        return obs
    env._client.observation = mock_obs
    
    async def mock_kill(tags):
        print("Killing tags:", tags)
    env._client.debug_kill_unit = mock_kill
    
    # Step 1: unit 1
    obs, reward, done, trunc, info = await env._async_step(1)
    print("Step 1 (unit 1) reward:", reward)
    
    # Step 2: unit 2
    obs, reward, done, trunc, info = await env._async_step(1)
    print("Step 2 (unit 2) reward:", reward)

if __name__ == "__main__":
    asyncio.run(test_env())
