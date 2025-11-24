"""
StarCraft 2 Bot Interface
Passes rich data including Effects to the hierarchical system
"""
from sc2.bot_ai import BotAI
from sc2.units import Units

class UnitControlBot(BotAI):
    def __init__(self, system, agent_id="main_agent"):
        super().__init__()
        self.system = system
        self.agent_id = agent_id
        self.agent = None
        self._previous_upgrades = set()

    async def on_start(self):
        print(f"Game started on {self.game_info.map_name}")
        if self.agent_id in self.system.unit_agents:
            self.agent = self.system.unit_agents[self.agent_id]
        else:
            self.agent = self.system.create_unit_agent(self.agent_id)

    async def on_step(self, iteration):
        # Robustly fetch units
        own_units = self.units if self.units else self.state.own_units
        enemy_units = self.enemy_units if self.enemy_units else self.state.enemy_units

        obs = {
            'units': own_units,
            'enemy_units': enemy_units,
            'effects': self.state.effects,
            'creep': self.state.creep,
            'game_info': self.game_info,
            'game_data': self._game_data,
        }

        self.system.process_observation(obs, player_id=1)

        actions = []
        if actions:
            await self.do_actions(actions)