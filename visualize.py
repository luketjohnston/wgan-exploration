
import matplotlib
matplotlib.use('tkagg')

import tensorflow as tf
import os
from tensorflow.keras import Model
import gym


import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

import agent
from wgan import *


# LOAD MNIST DATASET
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'gan.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')

gan = tf.saved_model.load(model_savepath)

# make environment
env = gym.make(agent.ENVIRONMENT)
state1 = env.reset()
state2 = env.step(env.action_space.sample())[0]
state3 = env.step(env.action_space.sample())[0]
state4 = env.step(env.action_space.sample())[0]
statelist = [state1, state2, state3, state4]
statelist = [tf.image.rgb_to_grayscale(s) for s in statelist]
statelist = [tf.image.resize(s,(84,110)) for s in statelist] #TODO does method of downsampling matter?
statelist = [s / 255.0 for s in statelist]
state = tf.stack(statelist, -1)
state = tf.squeeze(state)


while True:
  statelist.pop(0)
  observation = env.step(env.action_space.sample())[0]
  observation = tf.image.rgb_to_grayscale(observation)
  observation = tf.image.resize(observation,(84,110)) #TODO does method of downsampling matter?
  observation = observation / 255.0
  statelist.append(observation)
  state = tf.stack(statelist, -1)
  state = tf.squeeze(state)
  state = state[:,:,:DEPTH]
  state = tf.expand_dims(state, 0)


  fig, axes = plt.subplots(1,2)
  
  original = tf.squeeze(state)
  axes[0].imshow(original, cmap=cm.gray)

  image_approxs = tf.squeeze(gan.generate(1))
  print(image_approxs.shape)
  print(image_approxs)

  axes[1].imshow(image_approxs, cmap=cm.gray)
  #if MODULES == 1:
  #  axes[1].imshow(image_approxs, cmap=cm.gray)

  #else:
  #  for m in range(MODULES):
  #    axes[m+1].imshow(image_approxs[m], cmap=cm.gray)

  

  
  plt.show()
    




env.close()



