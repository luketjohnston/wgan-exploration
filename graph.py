import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

import agent
import wgan


with open(agent.picklepath, "rb") as f: 
  save = pickle.load(f)
  episode_rewards = save['episode_rewards']
  loss_policy = save['loss_policy']
  loss_value = save['loss_value']
  loss_entropy = save['loss_entropy']


y = loss_entropy

fig, [ax1,ax2,ax3,ax4] = plt.subplots(nrows=4)
ax1.plot(loss_policy)
ax2.plot(loss_value)
ax3.plot(episode_rewards)
ax4.plot(loss_entropy)

#ax.grid(True)
#start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
#ax.set_yscale('log')
plt.show()
