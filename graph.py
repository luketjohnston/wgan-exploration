import matplotlib.pyplot as plt
import argparse
import os
import pickle
import numpy as np

import agent
import wgan


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="explore with |wgan critic loss| intrinsic reward")
  parser.add_argument('load_model', type=str, help="path to model to load")
  args = parser.parse_args()
  
  savepath = args.load_model
  picklepath = os.path.join(savepath, 'save.pickle')
  
  
  with open(picklepath, "rb") as f: 
    save = pickle.load(f)
    episode_rewards = save['episode_rewards']
    loss_policy = save['loss_policy']
    loss_value = save['loss_value']
    loss_entropy = save['loss_entropy']
    loss_critic = save['critic_losses']
    loss_gen = save['gen_losses']
  
  
  y = loss_entropy
  
  fig, [ax1,ax2,ax3,ax4,ax5,ax6] = plt.subplots(nrows=6)
  ax1.plot(loss_policy)
  ax2.plot(loss_value)
  ax3.plot(episode_rewards)
  ax4.plot(loss_entropy)
  ax5.plot(loss_critic)
  ax6.plot(loss_gen)
  
  #ax.grid(True)
  #start, end = ax.get_ylim()
  #ax.set_ylim(0,0.1)
  #ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
  #ax.set_yscale('log')
  plt.show()
