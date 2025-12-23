
from sc2.client import Client
import inspect
try:
    print(inspect.signature(Client.observation))
except Exception as e:
    print(e)
