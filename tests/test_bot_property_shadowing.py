
from sc2.bot_ai import BotAI
from sc2.units import Units
import numpy as np

class ShadowBot(BotAI):
    def __init__(self, conn=None):
        # We don't call super().__init__ because it requires many arguments in some versions
        # We just want to see how it behaves
        self.units = Units([], self)

def test_shadowing():
    print(f"Attributes in BotAI: {[a for a in dir(BotAI) if not a.startswith('__')]}")
    
    bot = ShadowBot()
    print(f"Is 'units' in bot.__dict__? {'units' in bot.__dict__}")
    
    # Check if 'units' is a property in BotAI
    is_prop = isinstance(getattr(BotAI, "units", None), property)
    print(f"Is BotAI.units a property? {is_prop}")

if __name__ == "__main__":
    test_shadowing()
