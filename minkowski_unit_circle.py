import os

import matplotlib.pyplot as plt
import numpy as np

from betterment_colors import BETTERMENT_PALETTE

# Quick script to generate a plot of the unit circle in various Lebesgue spaces.

fig = plt.figure()
ax = plt.subplot(111)

for i, p in enumerate([0.5, 0.75, 1, 2, 5, 10]):
    steps = np.linspace(0, 2 * np.pi, 1000)
    xs = np.cos(steps)
    ys = (1 - (abs(xs) ** p)) ** (1 / p)
    ys[steps >= np.pi] *= -1
    plt.plot(xs, ys, label='{0}'.format(p), c=BETTERMENT_PALETTE[i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('images', 'minkowski_p_unit_circles.png'), dpi=300)
plt.close()
