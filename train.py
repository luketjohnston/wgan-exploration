import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random
import os
from tensorflow.keras import Model
import gym
from contextlib import ExitStack
from tf_agents.replay_buffers import tf_uniform_replay_buffer


# TODO: probably need to update experience replay frequencies, 
# right now all learning happens within first replay and subsequent
# replays remain constant.

LOAD_SAVE = True
if not LOAD_SAVE:
  tf.config.run_functions_eagerly(True)



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from agent import *

SAVE_CYCLES = 100
BATCH_SIZE = 16

CYCLES_BETWEEN_ACTING = 10000

CRITIC_BATCHES = 5
GEN_BATCHES = 1
#LR = 0.01

PRINT_GRADS = False
#PRINT_LOSS = 100

STEPS_BETWEEN_TRAINING = 1200 * 16
#STEPS_BETWEEN_TRAINING = 50*16




dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'gan.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')


# LOAD MNIST DATASET
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#train_ds = tf.data.Dataset.from_tensor_slices(
#    (x_train, y_train)).shuffle(10000).batch(32)
#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

if __name__ == '__main__':


  with open(loss_savepath, "rb") as f: 
    (criticLosses, genLosses) = pickle.load(f)
  
  
  if (LOAD_SAVE):
    gan = tf.saved_model.load(model_savepath)
  else:
    gan = GAN()
  
  
  #criticOpt = tf.keras.optimizers.Adam()
  criticOpt = tf.keras.optimizers.RMSprop()
  genOpt = tf.keras.optimizers.RMSprop()
  
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
  
    
  
    for c in range(CYCLES_BETWEEN_ACTING):
  
      for b in range(CRITIC_BATCHES):
        real_images = sample_batch(replay_buffer, BATCH_SIZE)
  
        with tf.GradientTape(watch_accessed_variables=False) as tape:
          tape.watch(gan.critic_vars)
  
          fake_images = gan.generate(BATCH_SIZE)
          criticLoss = gan.criticLoss(real_images, fake_images)
  
        grad = tape.gradient(criticLoss, gan.critic_vars)
        criticOpt.apply_gradients(zip(grad, gan.critic_vars))
  
        # FOR SOME REAson the default tf variable constraint isn't getting
        # saved and loaded. So just have to do it here
        #for v in gan.critic_vars:
        #  v.assign(tf.clip_by_value(v, -0.01,0.01))

        #print('grad maxes')
        #for g in grad:
        #  print(tf.reduce_max(g))
  
        criticLosses += [criticLoss.numpy()]
      
  
      for b in range(GEN_BATCHES):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
          tape.watch(gan.gen_vars)
   
          genLoss = gan.genLoss(BATCH_SIZE)
  
        grad = tape.gradient(genLoss, gan.gen_vars)
        genOpt.apply_gradients(zip(grad, gan.gen_vars))
  
        genLosses += [genLoss.numpy()]
     
      #criticLossStr = '[' + ''.join(['{:6f}, '.format(l) for l in criticLoss.numpy()]) + ']'
      criticLossStr = '{:6f}, '.format(criticLoss.numpy())
      #genLossStr = '[' + ''.join(['{:6f}, '.format(l) for l in genLoss.numpy()]) + ']'
      genLossStr = '{:6f}, '.format(genLoss[0].numpy())
      print(criticLossStr + genLossStr)
      
  
  
      if not c % SAVE_CYCLES:
        print('Saving model...')
        tf.saved_model.save(gan, model_savepath)
        with open(loss_savepath, "wb") as fp:
          pickle.dump((criticLosses, genLosses), fp)
      
  
    
  
  
  
  
  
  
  
  
  env.close()



