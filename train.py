import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import os
from tensorflow.keras import Model
import gym


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from stepped_encoder import *

SAVE = 100
BATCH_SIZE = 128

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


with open(loss_savepath, "rb") as f: 
  losses = pickle.load(f)


encoder = tf.saved_model.load(model_savepath)

# TODO should I save and load the optimizer?
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON)
opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
opt = tf.keras.optimizers.SGD(learning_rate=0.000001)



b = 0
while True:
  b += 1

  batchlist = []

  # Show what input looks like
  #original = tf.squeeze(state[:,:,0])
  #plt.imshow(original, cmap=cm.gray)
  #plt.show()

  #if b % 4000 == 3999:
  #  mystate = tf.expand_dims(state, 0)
  #  image = encoder.autoencode(mystate)
  #  image = image[:,:,:,0]
  #  image = tf.squeeze(image)
  #  print(image[44,:])
 
  #  # original image show first
  #  original = tf.squeeze(state[:,:,0])
  #  plt.imshow(original, cmap=cm.gray)
  #  #plt.show()

  #  plt.imshow(image, cmap=cm.gray)
  #  plt.show()

  for i in range(BATCH_SIZE):
    statelist.pop(0)
    observation = env.step(env.action_space.sample())[0]

    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation,(84,110)) #TODO does method of downsampling matter?

    statelist.append(observation)

    state = tf.stack(statelist, -1)
    state = tf.squeeze(state)
    batchlist.append(state)
    
  if b % SAVE == (SAVE - 1):
    print('Saving model...')
    tf.saved_model.save(encoder, model_savepath)
    with open(loss_savepath, "wb") as fp:
      pickle.dump(losses, fp)
    
    

  env.reset()
  batch = tf.stack(batchlist, 0)
  with tf.GradientTape() as tape:
    loss = encoder.loss(batch)
    all_vars = tape.watched_variables()
    gradients = tape.gradient(loss, all_vars)
    opt.apply_gradients(zip(gradients, all_vars))
  print(loss)
  losses += [loss]

    

  








env.close()



