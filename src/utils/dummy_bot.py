from sc2.bot_ai import BotAI

class DummyBot(BotAI):
    """A minimal bot that does nothing.
    This is used to satisfy python-sc2's requirement for a BotAI instance
    when creating a game, even if the environment is controlled externally.
    """
    async def on_step(self, iteration: int):
        pass
