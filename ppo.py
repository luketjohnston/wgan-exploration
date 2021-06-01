import tensorflow as tf
import multiprocessing as mp
import matplotlib
import timeit
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random
import pickle 
import os
from skimage.transform import resize
from tensorflow.keras import Model
import gym
from contextlib import ExitStack


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent
import wgan
from vecenv import VecEnv # not parallelized yet

from atari_wrappers import WarpFrame, NoopResetEnv, EpisodicLifeEnv, ClipRewardEnv, FrameStack, ScaledFloatFrame, MaxAndSkipEnv, FireResetEnv

if tf.config.list_physical_devices('GPU'):
  # For some reason this is necessary to prevent error
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


def makeEnv():
  env = gym.make(agent.ENVIRONMENT)
  if not agent.ENVIRONMENT == 'CartPole-v1' and not agent.ENVIRONMENT == 'Acrobot-v1':
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)

    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
      env = FireResetEnv(env)
    env = WarpFrame(env, 84, 84)
    env = ScaledFloatFrame(env) # negates optimization of LazyFrames in FrameStack
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
  return env


LOAD_SAVE = True
PROFILE = False

USE_WGAN = False
GAN_REWARD_WEIGHT = 0.01
SAVE_CYCLES = 20
ENVS = agent.ENVS
ROLLOUT_LEN = agent.ROLLOUT_LEN
MINIBATCH_SIZE = agent.MINIBATCH_SIZE
MINIBATCHES = ENVS * ROLLOUT_LEN // MINIBATCH_SIZE # must be multiple of 
FRAMES_PER_CYCLE = ENVS * ROLLOUT_LEN
EPOCHS = agent.EPOCHS
WGAN_TRAINING_CYCLES = 1
CRITIC_BATCHES = 20
GEN_BATCHES = 1

PARAM_UPDATES_PER_CYCLE = MINIBATCHES * EPOCHS


''' 
given vecenv and actor, 
computes the generalized advantage estimate for a rollout of n steps.

returns:
[states, actions, dones, returns value_ests, act_log_probs], episode_rewards

episode_rewards is a list of total reward for any episodes that terminated during
the rollout.
'''
def getRollout(vecenv, n, actor):

  # + 1 to hold extra state after last transition, to use to get final one-step td estimate
  states = np.zeros([n + 1, vecenv.nenvs] + agent.INPUT_SHAPE, np.float32)
  rewards = np.zeros([n, vecenv.nenvs])
  actions = np.zeros([n, vecenv.nenvs], dtype=np.int)
  dones = np.zeros([n, vecenv.nenvs], dtype=np.bool)

  act_probs = np.zeros([n,vecenv.nenvs])


  episode_scores = []

  states[0] = vecenv.getStates()

  for i in range(n):
    policy_logits = actor.policy(vecenv.getStates())
    actions[i], act_probs[i] = actor.act(policy_logits)
    states[i+1], rewards[i], dones[i], ep_scores = vecenv.step(actions[i])
    episode_scores += ep_scores

  return (states, rewards, actions, dones, act_probs), episode_scores

''' given rollout = (states, rewards, actions, dones, act_log_probs)
as returned by getRollout, n = length of rollout, and actor, 
returns the GAE adv estimates and returns,
as well as the rest of the inputs needed for the batch of training
'''

def getGAE(rollout, n, actor):

  (states, rewards, actions, dones, act_probs) = rollout
  #states = np.reshape(states, (states.shape[0] * states.shape[1],) + states.shape[2:])
  value_ests = tf.map_fn(actor.value, states)
  #value_ests = np.reshape(value_ests, (-1, n) + value_ests.shape[1:])

  advs = np.zeros((n,vecenv.nenvs))

  next_adv = 0.0

  # compute returns and advs using Generalized Advantage Estimation
  for t in reversed(range(n)):
    nonterminal = (1 - dones[t])
    delta = rewards[t] + agent.DISCOUNT * value_ests[t+1] * nonterminal - value_ests[t]
    advs[t] = delta + agent.LAMBDA * agent.DISCOUNT * next_adv * nonterminal 
    next_adv = advs[t]

  states = states[:-1,...]          # toss last state and value estimage
  value_ests = value_ests[:-1,...]  

  returns = advs + value_ests

  # reshape all the collected data into a single batch before returning
  batch = [states, actions, returns, value_ests, act_probs]
  for i,x in enumerate(batch):
    batch[i] = np.reshape(x, (n*vecenv.nenvs,) + x.shape[2:])

  return batch



if __name__ == '__main__':

  with open(agent.picklepath, "rb") as f: 
    agentsave = pickle.load(f)
    for x in ['loss_policy', 'loss_value', 'loss_entropy', 'episode_rewards']:
      if not x in agentsave:
        agentsave[x] = []
  
  
  if (LOAD_SAVE):
    actor = tf.saved_model.load(agent.model_savepath)
    if USE_WGAN:
      gan = tf.saved_model.load(wgan.model_savepath)
  else:
    actor = agent.Agent()
    if USE_WGAN: gan = wgan.WGan()

  vecenv = VecEnv(makeEnv, ENVS)
  
  cycle = 0

  for cycle in range(0, agent.FRAMES // FRAMES_PER_CYCLE):

    if cycle == 2 and PROFILE:
      tf.profiler.experimental.start('logdir')
    if cycle == 3 and PROFILE:
      tf.profiler.experimental.stop()

    rollout, episode_rewards = getRollout(vecenv, ROLLOUT_LEN, actor)
    #[states, rewards, actions, dones, act_log_probs] = rollout

    agentsave['episode_rewards'] += episode_rewards

    if len(agentsave['episode_rewards']) > 0:
      av_ep_rew = sum(agentsave['episode_rewards'][-20:]) / len(agentsave['episode_rewards'][-20:])
      print('Average episode reward: ' + str(av_ep_rew))
      print('Episode count: ' + str(len(agentsave['episode_rewards'])))
    

    print("Frames: %d" % ((cycle + 1) * ROLLOUT_LEN * ENVS))
    print("iterations " + str(cycle + 1))
    #print("Param updates: %d" % (cycle * PARAM_UPDATES_PER_CYCLE))

    #if USE_WGAN:
    #  print('training WGAN')
    #  for c in range(WGAN_TRAINING_CYCLES):
    #    for _ in range(CRITIC_BATCHES):
    #      real_images = next(dataiter)
    #      critic_loss = model.trainCritic(real_images, BATCH_SIZE)
    #      save['critic_losses'] += [critic_loss]
    #    for _ in range(GEN_BATCHES):
    #      gen_loss = model.trainGen(BATCH_SIZE)
    #      save['gen_losses'] += [gen_loss]
    #    loss_str = ''.join('{:6f}, '.format(lossv) for lossv in (critic_loss, gen_loss))
    #    print('critic, gen: ' + loss_str)
       
  
    print('training ppo')
    total_datapoints = MINIBATCH_SIZE * MINIBATCHES
    indices = np.arange(total_datapoints)
    # TODO: does it help sample complexity to recompute GAE at each epoch? some implementations do this (minimalRL), some dont (stable baselines 3)
    # this paper https://arxiv.org/pdf/2006.05990.pdf recommends recomputing them for each epoch
    # my current implementation of getGAE is really slow
    batch = getGAE(rollout, ROLLOUT_LEN, actor)
    for e in range(EPOCHS):

      np.random.shuffle(indices)
      for mb_start in range(0,total_datapoints,MINIBATCH_SIZE):
        mb_indices = indices[mb_start:mb_start + MINIBATCH_SIZE]
        inputs = [d[mb_indices,...] for d in batch]
  
        loss_pve, grads, r, adv = actor.train(*inputs)

        loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss_pve)
  
        agentsave['loss_policy'] += [loss_pve[0].numpy()]
        agentsave['loss_value'] += [loss_pve[1].numpy()]
        agentsave['loss_entropy'] += [loss_pve[2].numpy()]
        
    print(loss_str)

    #for v in actor.vars:
    #  tf.print(tf.reduce_sum(tf.abs(v)))
    #for g in grads:
    #  tf.print(tf.reduce_max(tf.abs(g)))
      
  
  
    cycle += 1
    if not cycle % SAVE_CYCLES:
      print('Saving model...')
      tf.saved_model.save(actor, agent.model_savepath)
      tf.saved_model.save(actor, agent.backup_savepath)
      with open(agent.picklepath, "wb") as fp:
        pickle.dump(agentsave, fp)
      
      with open(agent.picklepath_backup, "wb") as fp:
        pickle.dump(agentsave, fp)
  
    
  
  
  
