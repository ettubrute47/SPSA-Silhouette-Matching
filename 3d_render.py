import numpy as np
from plotoptix import TkOptiX

n = 1000000  # 1M points, better not try this with matplotlib
xyz = 3 * (np.random.random((n, 3)) - 0.5)  # random 3D positions
r = 0.02 * np.random.random(n) + 0.002  # random radii

plot = TkOptiX()
plot.set_data("my plot", xyz, r=r)
plot.show()
