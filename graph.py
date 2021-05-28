import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

import agent
import wgan


with open(agent.picklepath, "rb") as f: 
  save = pickle.load(f)
  y = save['loss_value']


fig, ax = plt.subplots()
ax.plot(y)
ax.grid(True)
start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
#ax.set_yscale('log')
plt.show()
