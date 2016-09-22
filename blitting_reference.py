import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import time

fig, ax = plt.subplots()
plt.show(block=False)
background = fig.canvas.copy_from_bbox(ax.bbox)
line, = ax.plot(np.random.randn(100))

tstart = time.time()
num_plots = 0
while time.time()-tstart < 5:
    fig.canvas.restore_region(background)
    line.set_ydata(np.random.randn(100))
    # ax.draw_artist(ax.patch)
    ax.draw_artist(line)
    fig.canvas.blit(ax.bbox)
    # fig.canvas.flush_events()
    num_plots += 1
print(num_plots/5)


# fig, ax = plt.subplots()
# line, = ax.plot(np.random.randn(100))
# fig.canvas.update()
# plt.show(block=False)

# tstart = time.time()
# num_plots = 0
# while time.time()-tstart < 5:
#     line.set_ydata(np.random.randn(100))
#     ax.draw_artist(ax.patch)
#     ax.draw_artist(line)
#     fig.canvas.update()
#     fig.canvas.flush_events()
#     num_plots += 1
# print(num_plots/5)
