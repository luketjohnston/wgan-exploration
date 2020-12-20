import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import os
from tensorflow.keras import Model
import gym


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from multi_encoder import *

SAVE = 400
BATCH_SIZE = 16

PRINT_GRADS = False


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
opt_bkgd  = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7)
opt_conv = tf.keras.optimizers.SGD(learning_rate=0.01)

#opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
#opt = tf.keras.optimizers.SGD(learning_rate=0.000001)



b = 0
while True:
  b += 1

  batchlist = []

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
  with tf.GradientTape() as tape_bkgd, tf.GradientTape() as tape_adj:
    tape_bkgd.watch(encoder.background_vars)
    tape_adj.watch(encoder.adjustment_vars)

    background_ms = encoder.encode_background(batch)
    adjustment_ms = encoder.encode_adjustment(batch)
    background_sample = encoder.sample(background_ms)
    adjustment_sample = encoder.sample(adjustment_ms)
    background = encoder.background(background_sample)

    background_loss = encoder.background_loss(background, batch)
    adjustment_loss = encoder.adjustment_loss(adjustment_sample, background, batch)

  grad_back = tape_bkgd.gradient(background_loss, encoder.background_vars)
  grad_adj = tape_adj.gradient(adjustment_loss, encoder.adjustment_vars)

  opt_bkgd.apply_gradients(zip(grad_back, encoder.background_vars))
  
  if (background_loss < 110) and PRINT_GRADS:
    opt_conv.apply_gradients(zip(grad_adj, encoder.adjustment_vars))
    for x in grad_adj:
      print(tf.reduce_max(x))

  print([background_loss, adjustment_loss])
  #print([background_loss])
  losses += [[background_loss, adjustment_loss]]

    

  








env.close()



