import subprocess
from itertools import product

lr = ["2e-3", "2e-4"]
wds = ["1e-3", "1e-4"]


for l, w in product(lr, wds):
  subprocess.run(f"python experiments/inception-model.py {l} {w}", shell=True)


