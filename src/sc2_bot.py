"""
StarCraft 2 Bot Interface
Passes rich data including Effects to the hierarchical system.
Fixed:
1. Robust action execution using 'client.actions'.
2. Correct 'all_units' passing for observation.
3. Cleaned up debug random actions (Agent logic should be inserted where marked).
"""
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2

class UnitControlBot(BotAI):
    def __init__(self, system, agent_id="main_agent"):
        super().__init__()
        self.system = system
        self.agent_id = agent_id
        self.agent = None
        self._previous_upgrades = set()

    async def on_start(self):
        print(f"[Bot] Game started on {self.game_info.map_name}")

        # Log Map Dimensions for confirmation
        print(f"[Bot] Map Dimensions: {self.game_info.map_size.width}x{self.game_info.map_size.height}")
        area = self.game_info.playable_area
        print(f"[Bot] Playable Area: {area.width}x{area.height} (x:{area.x}, y:{area.y})")

        if self.agent_id in self.system.unit_agents:
            self.agent = self.system.unit_agents[self.agent_id]
        else:
            self.agent = self.system.create_unit_agent(self.agent_id)

    async def on_step(self, iteration):
        # 1. Filter Units Explicitly
        # Move Beacons from 'own_units' to 'enemy_units' if they snuck in
        raw_own = self.all_units
        raw_enemy = self.all_enemy_units

        real_own = []
        real_enemy = []

        # Helper to identify beacons
        def is_beacon(u):
            return "BEACON" in u.name.upper()

        # Process Allied List
        for u in raw_own:
            if is_beacon(u):
                real_enemy.append(u)  # Force Beacon to Enemy List
            else:
                real_own.append(u)

        # Process Enemy List
        for u in raw_enemy:
            real_enemy.append(u)

        # 2. Construct Observation
        obs = {
            'units': real_own,
            'enemy_units': real_enemy,
            'effects': self.state.effects,
            'creep': self.state.creep,
            'game_info': self.game_info,
            'game_data': self._game_data,
        }

        self.system.process_observation(obs, player_id=1)

        actions = []

        # --- INSERT AGENT INFERENCE LOGIC HERE ---
        # Example if you have a trained model:
        # Example if you have a trained model:
        # action_idx = model.predict(obs)
        # world_x, world_y = self.agent.convert_spatial_action_to_world(action_idx)
        # cmd = unit(AbilityId.MOVE, Point2((world_x, world_y)))
        # actions.append(cmd)
        # -----------------------------------------

        if actions:
            # Filter valid commands and execute
            valid_actions = [a for a in actions if hasattr(a, 'combining_tuple')]
            if valid_actions:
                await self.client.actions(valid_actions)