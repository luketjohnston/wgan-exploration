import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from train import *

dir_path = os.path.dirname(os.path.realpath(__file__))
loss_savepath = os.path.join(dir_path, 'loss.pickle')


with open(loss_savepath, "rb") as f: 
  (criticLosses, genLosses) = pickle.load(f)


fig, ax = plt.subplots()
ax.plot(np.arange(0, len(genLosses), GEN_BATCHES / CRITIC_BATCHES), criticLosses)
ax.plot(genLosses)
ax.grid(True)
start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
#ax.set_yscale('log')
plt.show()
