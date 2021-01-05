import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
loss_savepath = os.path.join(dir_path, 'loss.pickle')


with open(loss_savepath, "rb") as f: 
  losses = pickle.load(f)


fig, ax = plt.subplots()
ax.plot(losses)
ax.grid(True)
start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
ax.set_yscale('log')
plt.show()
