
from train.envs.level_3_env import Level3Env
from unittest.mock import MagicMock

def test_init():
    try:
        env = Level3Env(level_name="level_move_multi_0", num_workers=4, training_conf={"config_dir": "config_universal"})
        print("action_heads:", env.action_heads)
        print("action_space:", env.action_space)
        if hasattr(env.action_space, 'nvec'):
            print("nvec:", env.action_space.nvec)
            print("sum:", sum(env.action_space.nvec))
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_init()
