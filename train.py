import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random
import os
from tensorflow.keras import Model
import gym
from tf_agents.replay_buffers import tf_uniform_replay_buffer


# TODO: probably need to update experience replay frequencies, 
# right now all learning happens within first replay and subsequent
# replays remain constant.



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from agent import *

SAVE = 400
BATCH_SIZE = 16
#LR = 0.01

EPOCHS = 50

PRINT_GRADS = False
#PRINT_LOSS = 100

STEPS_BETWEEN_TRAINING = 400 * 16
TRAINING_BATCHES = 4000




dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'encoder.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')

# LOAD MNIST DATASET
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#train_ds = tf.data.Dataset.from_tensor_slices(
#    (x_train, y_train)).shuffle(10000).batch(32)
#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


with open(loss_savepath, "rb") as f: 
  losses = pickle.load(f)


encoder = tf.saved_model.load(model_savepath)

#opt_conv = tf.keras.optimizers.Adam(learning_rate=LR)
opt_bkgd = tf.keras.optimizers.Adam()
#opt_adj = tf.keras.optimizers.Adam(learning_rate=LR)
opt_adj = tf.keras.optimizers.Adam()

# make environment
env = gym.make('MontezumaRevenge-v0')
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





# make replay buffer
#data_spec = (
#  tf.TensorSpec([WIDTH,HEIGHT,DEPTH], tf.float32),
#  )
#max_length = 1000
#
#replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#  data_spec,
#  batch_size=BATCH_SIZE,
#  max_length=max_length)



def sample_batch(replay_buffer, batch_size):
  sample = random.sample(replay_buffer, batch_size)
  return tf.stack(sample, 0)




while True: 


  replay_buffer = []

  for b in range(STEPS_BETWEEN_TRAINING):
    statelist = statelist[1:]
    observation, reward, done, info = env.step(env.action_space.sample())
    if (done): env.reset()
    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation,(84,110)) #TODO does method of downsampling matter?
    observation = observation / 255.0
    statelist.append(observation)
    state = tf.stack(statelist, -1)
    state = tf.squeeze(state)

    state = state[:,:,:DEPTH]



        
    replay_buffer.append(state)

    #fig, ax1 = plt.subplots(1,1)
    #ax1.imshow(batch[0,:,:,0], cmap=cm.gray)
    #plt.show()

  
  print('starting experience replay...')

  for b in range(TRAINING_BATCHES):

    batch = sample_batch(replay_buffer, BATCH_SIZE)

    with tf.GradientTape() as tape_bkgd, tf.GradientTape() as tape_adj:
      tape_bkgd.watch(encoder.background_vars)
      tape_adj.watch(encoder.adjustment_vars)

      background = encoder.background(batch)
      encoding = encoder.encode_adjustment(batch)
      adjustment = encoder.decode_adjustment(encoding)
      #background_sample = encoder.sample(background_ms)
      #adjustment_sample = encoder.sample(adjustment_ms)

      background_loss = encoder.background_loss(background, batch)
      adjustment_loss = encoder.adjustment_loss(background, adjustment, batch)

    grad_back = tape_bkgd.gradient(background_loss, encoder.background_vars)
    grad_adj = tape_adj.gradient(adjustment_loss, encoder.adjustment_vars)

    opt_bkgd.apply_gradients(zip(grad_back, encoder.background_vars))
    opt_adj.apply_gradients(zip(grad_adj, encoder.adjustment_vars))


    losses += [adjustment_loss]
    print(adjustment_loss)


  print('finished experience replay, saving model')
  tf.saved_model.save(encoder, model_savepath)
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
    

  








env.close()



