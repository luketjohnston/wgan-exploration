import tensorflow as tf
import timeit
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random
import pickle
import os
from skimage.transform import resize
from tensorflow.keras import Model
import gym
from contextlib import ExitStack



LOAD_SAVE = True
#if not LOAD_SAVE:
#  tf.config.run_functions_eagerly(True)
#else:
#  # TODO figure out graph mode
#  # seems like eager is default for me, have to manually disable
#  #tf.compat.v1.disable_eager_execution() # TF2; above holds
#  pass


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent
import wgan

GAN_REWARD_WEIGHT = 0.01
#LR = 0.00001 use this for fully connected
LR = 0.001 # this is tf default
SAVE_CYCLES = 1
BATCH_SIZE = 16
ENVS = 16
#STEPS_BETWEEN_TRAINING = 1200 * 16
STEPS_BETWEEN_TRAINING = 128 * 8
#STEPS_BETWEEN_TRAINING = 128
EPOCHS = 3
WGAN_TRAINING_CYCLES = 100
CRITIC_BATCHES = 5
GEN_BATCHES = 1



if __name__ == '__main__':

  #sess = tf.compat.v1.Session()
  #with sess.as_default():
  if True:

    with open(agent.loss_savepath, "rb") as f: 
      agentloss = pickle.load(f)
    with open(wgan.loss_savepath, "rb") as f: 
      wganloss = pickle.load(f)
    with open(agent.rewards_savepath, "rb") as f:
      episode_rewards = pickle.load(f)
    
    
    if (LOAD_SAVE):
      #actor = tf.saved_model.load(agent.model_savepath)
      actor = tf.saved_model.load(agent.model_savepath)
      #sess.run(tf.compat.v1.global_variables_initializer())
      gan = tf.saved_model.load(wgan.model_savepath)
    else:
      actor = agent.Agent()
      gan = wgan.WGan()
    
    agentOpt = tf.keras.optimizers.RMSprop(learning_rate = LR)
    criticOpt = tf.keras.optimizers.RMSprop()
    genOpt = tf.keras.optimizers.RMSprop()
    
    envs = []
    states = []
    statelists = []

    # make environment
    for i in range(ENVS):
 
      env = gym.make(agent.ENVIRONMENT)
      state1 = env.reset()
      # 0 action is 'NOOP'
      state2 = env.step(0)[0]
      state3 = env.step(0)[0]
      state4 = env.step(0)[0]
      statelist = [state1, state2, state3, state4]
      statelist = [tf.image.rgb_to_grayscale(s) for s in statelist]
      statelist = [tf.image.resize(s,(84,110)) for s in statelist] #TODO does method of downsampling matter?
      statelist = [s / 255.0 for s in statelist]
      state = tf.stack(statelist, -1)
      state = tf.squeeze(state)
      envs += [env]
      states.append(state)
      statelists += [statelist]

    states = tf.stack(states)
    
    
    
    cycle = 0
    agent_losses = []
    criticLosses = []
    genLosses = []
    total_rewards = [0 for _ in envs]

    #def sample_batch(replay_buffer, batch_size):
    #  sample = random.sample(replay_buffer, batch_size)
    #  return tf.stack(sample, 0)

    wgan_replay_buffer = []


    while True: 

      #starttime = timeit.default_timer()
      network_time = 0
      steptime = 0
      statelist_updates = 0
      update_lists = 0

      states_l, actions_l, old_action_probs_l, rewards_l, dones_l = [],[],[],[],[]
      print('acting')

      for step in range(STEPS_BETWEEN_TRAINING):
        #tmp1 = timeit.default_timer()
        policy_logits, values = actor.policy_and_value(states)
        actions, old_action_probs = actor.act(policy_logits)
        #tmp2 = timeit.default_timer()
        #network_time += tmp2 - tmp1
        #actions = sess.run(actions)

        next_states, rewards, dones = [], [], []
        #print('step ' + str(step))

        for i in range(ENVS):

          #tmp1 = timeit.default_timer()
          observation, reward, done, info = envs[i].step(actions[i])
          #tmp2 = timeit.default_timer()
          #steptime += tmp2 - tmp1
          
          total_rewards[i] += reward
          if (done): 
            #tmp1 = timeit.default_timer()
            envs[i].reset()
            #tmp2 = timeit.default_timer()
            #steptime += tmp2 - tmp1
            episode_rewards += [total_rewards[i]]
            print("Finished episode %d, reward: %f" % (len(episode_rewards), total_rewards[i]))
            total_rewards[i] = 0
         
          #tmp1 = timeit.default_timer()

          observation = tf.image.rgb_to_grayscale(observation)
          observation = tf.image.resize(observation,(84,110)) 
          #tmp2 = timeit.default_timer()
          observation = observation / 255.0
          #network_time += tmp2 - tmp1

     

          #tmp1 = timeit.default_timer()
          statelists[i] = statelists[i][1:]
          statelists[i].append(observation)
          state = tf.stack(statelists[i], -1)
          state = tf.squeeze(state)

          next_states += [state]
          rewards += [reward]
          dones += [float(done)]
          #tmp2 = timeit.default_timer()
          #statelist_updates += tmp2 - tmp1

        #tmp1 = timeit.default_timer()

        # need to copy to CPU so we don't use all the GPU memory
        #with tf.device('/device:CPU:0'):
        if True:
          states_l.append(tf.identity(states))
          actions_l.append(tf.identity(actions))
          old_action_probs_l.append(tf.identity(old_action_probs))
          rewards_l.append(tf.squeeze(tf.stack(rewards)))
          dones_l.append(tf.stack(dones))

        states = tf.stack(next_states)
        #tmp2 = timeit.default_timer()
        #update_lists += tmp2 - tmp1

        #print('total time: %f' % (else_time - start))
        #print('else time: %f' % (else_time - network_time))
        #print('obs time: %f' % (obs2time - obs2time))


      #endtime = timeit.default_timer()
      #print('network time: %f' % (network_time))
      #print('step time: %f' % (steptime))
      #print('statelist time: %f' % (statelist_updates))
      #print('update lists time: %f' % (update_lists))
      #print('total time: %f' % (endtime - starttime))
      #print('values from one step:')
      #print(values)

      # go through rewards and add wgan reward
      for i in range(len(rewards_l)):
        rewards_l[i] += GAN_REWARD_WEIGHT * tf.abs(gan.critic(states_l[i][:,:,:,:1])[0] * (1 - dones_l[i]))

      # compute value targets (discounted returns to end of episode (or end of training))
      rewards_l[-1] += (1 - dones_l[-1]) * actor.policy_and_value(states)[1]
      #with tf.device('/device:CPU:0'):
      if True:
        for i in range(len(rewards_l)-2, -1, -1):
          rewards_l[i] = rewards_l[i] + agent.DISCOUNT * (1-dones_l[i]) * rewards_l[i+1]

      # TODO do we need wgan replay buffer or can we just use states?

      print("Frames: %d" % (cycle * STEPS_BETWEEN_TRAINING * ENVS))
      print('training WGAN')

      for c in range(WGAN_TRAINING_CYCLES):
        for _ in range(CRITIC_BATCHES):
          real_images = random.sample(states_l, 1)[0][:,:,:,:1]
          with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(gan.critic_vars)
            fake_images = gan.generate(BATCH_SIZE)
            criticLoss = gan.criticLoss(real_images, fake_images)
          grad = tape.gradient(criticLoss, gan.critic_vars)
          criticOpt.apply_gradients(zip(grad, gan.critic_vars))
          criticLosses += [criticLoss.numpy()]
        for _ in range(GEN_BATCHES):
          with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(gan.gen_vars)
            genLoss = gan.genLoss(BATCH_SIZE)
          grad = tape.gradient(genLoss, gan.gen_vars)
          genOpt.apply_gradients(zip(grad, gan.gen_vars))
          genLosses += [genLoss.numpy()]
       
        criticLossStr = '{:6f}, '.format(criticLoss.numpy())
        genLossStr = '{:6f}, '.format(genLoss[0].numpy())
        print('Critic and gen loss: ' + criticLossStr + genLossStr)
    
      print('training ppo')
      indices = list(range(len(states_l)))
      for e in range(EPOCHS):
        print('epoch ' + str(e))
        random.shuffle(indices)
        for i in indices:
        #for states, actions, old_action_probs, target_values in datapoints:
          #print(i)
          states, actions, old_action_probs, target_values = states_l[i], actions_l[i], old_action_probs_l[i], rewards_l[i]
    
          with tf.GradientTape(watch_accessed_variables=True) as tape:
    
            loss_pve = actor.loss(states, actions, old_action_probs, target_values)
            #print("{:6f}, {:6f}, {:6f}".format(loss_pve[0].numpy(), loss_pve[1].numpy(), loss_pve[2].numpy()))

            loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss_pve)
            #input('input') 
            
    
          grad = tape.gradient(loss_pve, actor.vars)
          agentOpt.apply_gradients(zip(grad, actor.vars))
          #for v in grad:
          #  print(v)
    

    
          agent_losses += [loss_pve]
          
        print(loss_str)
         
        
    
    
      cycle += 1
      if not cycle % SAVE_CYCLES:
        print('Saving model...')
        tf.saved_model.save(actor, agent.model_savepath)
        tf.saved_model.save(gan, wgan.model_savepath)
        with open(agent.loss_savepath, "wb") as fp:
          pickle.dump(agent_losses, fp)
        with open(wgan.loss_savepath, "wb") as fp:
          pickle.dump((criticLosses, genLosses), fp)
        with open(agent.rewards_savepath, "wb") as fp:
          pickle.dump(episode_rewards, fp)

      
  
    
  
  
  
