
from sc2.sc2process import SC2Process
import inspect

print("Inspect SC2Process:")
print(dir(SC2Process))

try:
    p = SC2Process()
    print("Instance attributes:")
    print(dir(p))
except Exception as e:
    print(f"Could not instantiate SC2Process: {e}")
