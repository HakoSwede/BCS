import os

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

for p in [0.5, 0.75, 1, 2, 5, 10]:
    steps = np.linspace(0, 2*np.pi, 1000)
    xs = np.cos(steps)
    ys = (1 - (abs(xs) ** p))**(1 / p)
    ys[steps >= np.pi] *= -1
    plt.plot(xs, ys, label='{0}'.format(p))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('images', 'minkowski_p_unit_circles.png'), dpi=300)
plt.close()