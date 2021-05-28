import os
import tensorflow as tf
import random
from tensorflow.keras import Model
import gym
import multiprocessing as mp
import pickle
from queue import Queue

import wgan

if tf.config.list_physical_devices('GPU'):
  # For some reason this is necessary to prevent error
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


LOAD_SAVE = True
PROFILE = True
CPU_CORES = 6

ENVS = 128

SAVE_CYCLES = 50
BATCH_SIZE = 128 
SAVE_STATS_EVERY = 50

CRITIC_BATCHES = 20
GEN_BATCHES = 1

def preprocess(s):
  s = tf.image.rgb_to_grayscale(s)
  s = tf.image.resize(s,(wgan.INPUT_SHAPE[0],wgan.INPUT_SHAPE[1]))
  #s = tf.squeeze(s)
  return tf.cast(s / 255.0, tf.float32)

def getDataProcess(dataQueue):
  with tf.device('/device:CPU:0'):
    env = wgan.makeEnv()
    state1 = tf.cast(env.reset(), tf.float32)
    observation = preprocess(state1)
    while True:
      state,reward,done,info = env.step(env.action_space.sample())
      if done:     
        state = env.reset()
      state = preprocess(state)
      dataQueue.put(state)


if __name__ == '__main__':

  # need to spawn processes so they don't copy tensorflow state and try to do stuff on GPU
  mp.set_start_method('spawn')

  if True:

    with open(wgan.picklepath, "rb") as f: 
      save = pickle.load(f)
      for x in ['critic_losses', 'gen_losses']:
        if not x in save:
          save[x] = []
      SAVE_STATS_EVERY = SAVE_STATS_EVERY if not 'save_stats_every' in save else save['save_stats_every']
    
    if (LOAD_SAVE):
      model = tf.saved_model.load(wgan.model_savepath)
        
    else:
      model = wgan.WGAN()
    
    states = []
    statelists = []



    # make environment
    env = wgan.makeEnv()
    state1 = tf.cast(env.reset(), tf.float32)
    # 0 action is 'NOOP'
    state2 = tf.cast(env.step(0)[0], tf.float32)
    state3 = tf.cast(env.step(0)[0], tf.float32)
    state4 = tf.cast(env.step(0)[0], tf.float32)
    statelist = [state1, state2, state3, state4]
    statelist = [preprocess(s) for s in statelist]
    state = tf.stack(statelist, -1)
    observation = statelist[-1] # most recent observation

    cycle = 0
    total_rewards = 0
    

    saved_data_shape = tuple(wgan.INPUT_SHAPE[:-1])

    # TODO: should I convert all these to numpy arrays? probably...
    states_l  = statelist[:-1]
    actions_l = [0 for _ in statelist[:-1]]
    rewards_l = [0 for _ in statelist[:-1]]
    dones_l   = [0 for _ in statelist[:-1]]




    dataQueue = mp.Queue(maxsize = ENVS * 4) # no particular reason for this size. Want to be able to buffer a few new datapoints if possible


    def datasetGenerator():
      while True:
        data = dataQueue.get()
        yield data

    output_sig = tf.TensorSpec(wgan.INPUT_SHAPE, dtype=wgan.FLOAT_TYPE)
    dataset = tf.data.Dataset.from_generator(datasetGenerator, output_signature=output_sig)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
    dataiter = iter(dataset)


    # disable cuda on child processes
    os.environ['CUDA_VISIBLE_DEVICES'] = '';
    sample_procs = []
    for i in range(ENVS):
      print('starting ' + str(i))
      sample_procs.append(mp.Process(target = getDataProcess, args=(dataQueue,)))
      sample_procs[-1].daemon = True
      sample_procs[-1].start()
    print('started all procs')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0';

    try: # need to catch KeyboardInterrupt, to handle terminating processes

      cycle = 0
      while True: 
        if cycle == 1 and PROFILE:
          tf.profiler.experimental.start('logdir')
        if cycle == 10 and PROFILE:
          tf.profiler.experimental.stop()

          
        for _ in range(CRITIC_BATCHES):
          real_images = next(dataiter)
          critic_loss = model.trainCritic(real_images, BATCH_SIZE)
          save['critic_losses'] += [critic_loss]
        for _ in range(GEN_BATCHES):
          gen_loss = model.trainGen(BATCH_SIZE)
          save['gen_losses'] += [gen_loss]

        loss_str = ''.join('{:6f}, '.format(lossv) for lossv in (critic_loss, gen_loss))
        print(loss_str)


        cycle += 1
        if not cycle % SAVE_CYCLES:
          print('Cycle number: ' + str(cycle))
          print('Saving model...')
          tf.saved_model.save(model, wgan.model_savepath)
          with open(wgan.picklepath, "wb") as fp:
            save['save_stats_every'] = SAVE_STATS_EVERY
            pickle.dump(save, fp)

         
    except KeyboardInterrupt:
      for p in sample_procs:
        p.terminate()
      raise KeyboardInterrupt
    

  
  
