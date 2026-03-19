
from sc2.bot_ai import BotAI
import inspect

def inspect_bot_ai():
    print(f"File: {inspect.getfile(BotAI)}")
    
    # Check for 'units' property in BotAI or its parents
    for cls in inspect.getmro(BotAI):
        print(f"Checking class: {cls}")
        if 'units' in cls.__dict__:
            attr = cls.__dict__['units']
            print(f"Found 'units' in {cls}: {type(attr)}")
            if isinstance(attr, property):
                 print(f"  fget: {attr.fget}")
        
    # Check for 'all_units'
    for cls in inspect.getmro(BotAI):
        if 'all_units' in cls.__dict__:
            attr = cls.__dict__['all_units']
            print(f"Found 'all_units' in {cls}: {type(attr)}")

if __name__ == "__main__":
    inspect_bot_ai()
