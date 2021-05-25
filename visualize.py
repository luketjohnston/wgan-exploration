
import matplotlib
matplotlib.use('tkagg')

import tensorflow as tf
import os
from tensorflow.keras import Model
import gym


import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

import wgan

with tf.device('/device:CPU:0'):

  gan = tf.saved_model.load(wgan.model_savepath)
  
  # make environment
  env = wgan.makeEnv()
  state1 = env.reset()
  
  
  while True:
    observation = env.step(env.action_space.sample())[0]
    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation,(84,110)) #TODO does method of downsampling matter?
    observation = observation / 255.0
  
    fig, axes = plt.subplots(1,2)
    
    original = tf.squeeze(observation)
    axes[0].imshow(original, cmap=cm.gray)
  
    image_approxs = tf.squeeze(gan.generate(1))
  
    axes[1].imshow(image_approxs, cmap=cm.gray)
  
    
    plt.show()
      
  
  
  
  
  env.close()
  
  
  
