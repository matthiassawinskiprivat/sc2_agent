
import inspect
from sc2.sc2process import SC2Process

try:
    print(inspect.getsource(SC2Process))
except Exception as e:
    print(e)
