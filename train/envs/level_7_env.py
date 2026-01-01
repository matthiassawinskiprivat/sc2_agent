from train.envs.level_3_env import Level3Env
from src.level_7_bot import Level7Bot
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
import numpy as np

class Level7Env(Level3Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure action space reflects Move + Attack (2 * GridSize^2)
        # Level0Env init sets action_space to Discrete(GridSize^2)
        # We need to update it.
        # Note: We must import spaces inside or check if it's already set
        from gymnasium import spaces
        grid_area = self.grid_size * self.grid_size
        self.action_space = spaces.Discrete(2 * grid_area)
        
    def _log(self, message):
        super()._log(f"[Level7Env] {message}")

    async def _setup_game_session(self):
        # Override to use Level7Bot and Terran Race for Player 1
        try:
             await self._client.ping()
        except Exception as e:
             self._log(f"Initial ping failed: {e}")
             raise e

        game_map = self._resolve_map(self.map_name)
        if self._bot is None:
            self._log("Creating Level7Bot...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            
            self._bot = Level7Bot(conn=DummyConn(), training_conf=self.training_conf)
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        self._log("Creating game (Terran vs Zerg)...")
        # Player 1 = Terran (Marine), Player 2 = Zerg (Zergling)
        await self._controller.create_game(
            game_map,
            [Bot(Race.Terran, self._bot), Computer(Race.Zerg, Difficulty.Easy)],
            realtime=self.visualize,
        )
        await self._client.join_game(race=Race.Terran)
        self._bot._generated_frame = -1
        game_info = await self._client.get_game_info()
        game_data = await self._client.get_game_data()

        try:
            from sc2.unit import UnitGameData
            UnitGameData._game_data = game_data
            UnitGameData._bot_object = self._bot
            self._bot.client = self._client
            self._bot._game_info = game_info
            self._bot._game_data = game_data
        except ImportError:
            pass
        
        self._bot._prepare_start(client=self._client, player_id=1, game_info=game_info, game_data=game_data)
        initial_observation = await self._client.observation()
        
        from sc2.game_state import GameState
        state = GameState(initial_observation.observation)
        
        # Standard wrapper (redefined locally or imported)
        # We can assume ProtoWrapper is available from super() scope imports or re-import
        from train.envs.level_0_env import ProtoWrapper
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

        # Initial Step
        try:
            await self._bot.on_step(0)
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        # Call super reset but we need to intercept to ensure Fog is disabled
        # Level3Env.reset calls Level0Env.reset which calls _reset_bot_state -> reset_game_state
        
        # We need to ensure debug_show_map is called AFTER game start/reset
        obs, info = super().reset(seed, options)
        
        # Disable Fog of War
        if self._client:
             self._loop.run_until_complete(self._client.debug_show_map())
        
        # Re-fetch obs after map reveal? Usually takes a step.
        # But 'reset_game_state' in Bot likely already spawned units.
        # Let's trust that the next step will show them.
        
        # Reset active tags logic for Marines
        if self._bot:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_units()]
             self.current_unit_index = 0
             if self.active_unit_tags:
                 obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
        
        return obs, info

    async def _async_step(self, action):
        """
        Modified step logic for Level 7 (Marines vs Zerglings).
        """
        # Ensure active tags are Marines
        if not hasattr(self, "active_unit_tags") or not self.active_unit_tags:
             self.active_unit_tags = [u.tag for u in self._bot.get_active_units()]
        
        if not self.active_unit_tags:
             # All marines dead?
             return np.zeros(self.observation_space.shape), 0, True, False, {}

        current_tag = self.active_unit_tags[self.current_unit_index]
        await self._bot.apply_action(current_tag, action)
        
        self.current_unit_index += 1
        
        if self.current_unit_index < len(self.active_unit_tags):
            # Intermediate unit step
            next_tag = self.active_unit_tags[self.current_unit_index]
            obs, _ = self._bot.get_observation_for_unit(next_tag)
            return obs, 0.0, False, False, {}
        else:
            # End of frame
            actions = getattr(self._bot, "actions", [])
            if actions:
                await self._client.actions(actions)
                self._bot.actions.clear()

            await self._client.step()
            
            response = await self._client.observation()
            from sc2.game_state import GameState
            state = GameState(response.observation)
            from train.envs.level_0_env import ProtoWrapper
            self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))

            # Rewards
            combat_reward = self._bot.calculate_combat_reward()
            step_penalty = self.training_conf.get('rewards', {}).get('step_penalty', -0.01)
            
            total_reward = combat_reward + step_penalty
            
            # Check Done
            enemies = self._bot.enemy_units(UnitTypeId.ZERGLING).amount
            marines = self._bot.units(UnitTypeId.MARINE).amount
            
            done = False
            if enemies == 0:
                done = True
                total_reward += self.training_conf.get('rewards', {}).get('win_reward', 50.0)
            elif marines == 0:
                done = True
                total_reward += self.training_conf.get('rewards', {}).get('loss_penalty', -10.0)
                
            # Reset Cycle
            self.active_unit_tags = [u.tag for u in self._bot.get_active_units()]
            self.current_unit_index = 0
            
            if self.active_unit_tags:
                 obs, _ = self._bot.get_observation_for_unit(self.active_unit_tags[0])
            else:
                 obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
                 
            return obs, total_reward, done, False, {}
