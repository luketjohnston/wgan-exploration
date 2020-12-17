import tensorflow as tf
import os
from tensorflow.keras import Model
import gym


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from agent import *


SHOW_STEP = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'encoder.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')

env = gym.make('MontezumaRevenge-v0')

state1 = env.reset()
state2 = env.step(env.action_space.sample())[0]
state3 = env.step(env.action_space.sample())[0]
state4 = env.step(env.action_space.sample())[0]
statelist = [state1, state2, state3, state4]

statelist = [tf.image.rgb_to_grayscale(s) for s in statelist]
statelist = [tf.image.resize(s,(84,110)) for s in statelist] #TODO does method of downsampling matter?

state = tf.stack(statelist, -1)
state = tf.squeeze(state)

encoder = tf.saved_model.load(model_savepath)


i = 0
while (True):
  i += 1

  batchlist = []


  statelist.pop(0)
  observation = env.step(env.action_space.sample())[0]
  observation = tf.image.rgb_to_grayscale(observation)
  observation = tf.image.resize(observation,(84,110)) #TODO does method of downsampling matter?
  statelist.append(observation)

  state = tf.stack(statelist, -1)
  state = tf.squeeze(state)
  batchlist.append(state)
    
  if i % SHOW_STEP == 0:

    fig, (ax1,ax2) = plt.subplots(1,2)
    
    # Show what input looks like
    original = tf.squeeze(state[:,:,0])
    ax1.imshow(original, cmap=cm.gray)

    mystate = tf.expand_dims(state, 0)
    print('mystate shape')
    print(mystate.shape)
    image = encoder.autoencode(mystate)
    print('autoencoder shape')
    print(image.shape)
    image = image[:,:,:,0]
    image = tf.squeeze(image)
    print(image[44,:])
 
    ax2.imshow(image, cmap=cm.gray)
    plt.show()
    




env.close()



