
import matplotlib
matplotlib.use('tkagg')

import tensorflow as tf
import os
from tensorflow.keras import Model
import gym
import argparse


import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

import wgan



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="explore with |wgan critic loss| intrinsic reward")
  parser.add_argument('load_model', type=str, help="path to model to load")
  args = parser.parse_args()

  savepath = args.load_model
  wgan_savepath = os.path.join(savepath, 'wgan')
  actor_savepath = os.path.join(savepath, 'actor')

  with tf.device('/device:CPU:0'):
  
    #actor = tf.saved_model.load(actor_savepath)
    wgan = tf.saved_model.load(wgan_savepath)
    
    
    while True:
      fig, axes = plt.subplots(1,2)
      image_approxs = tf.squeeze(wgan.generate(2))
      axes[0].imshow(image_approxs[0], cmap=cm.gray)
      axes[1].imshow(image_approxs[1], cmap=cm.gray)
      plt.show()
        
    
    
    
    
    env.close()
  
  
  
