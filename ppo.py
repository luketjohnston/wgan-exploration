import tensorflow as tf
import multiprocessing as mp
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

from atari_wrappers import WarpFrame, NoopResetEnv, EpisodicLifeEnv, ClipRewardEnv, FrameStack, ScaledFloatFrame, MaxAndSkipEnv, FireResetEnv

if tf.config.list_physical_devices('GPU'):
  # For some reason this is necessary to prevent error
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


def makeEnv():
  env = gym.make(agent.ENVIRONMENT)
  if not agent.ENVIRONMENT == 'CartPole-v1':
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
#BATCH_SIZE = 16
ENVS = agent.ENVS
ROLLOUT_LEN = agent.ROLLOUT_LEN
MINIBATCH_SIZE = agent.MINIBATCH_SIZE
MINIBATCHES = ENVS * ROLLOUT_LEN // MINIBATCH_SIZE # must be multiple of 
EPOCHS = agent.EPOCHS
WGAN_TRAINING_CYCLES = 1
CRITIC_BATCHES = 20
GEN_BATCHES = 1

FRAMES_PER_CYCLE = ENVS * ROLLOUT_LEN

PARAM_UPDATES_PER_CYCLE = MINIBATCHES * EPOCHS

def actorProcess(env, state, cumulative_reward, actor):
  states = np.zeros([ROLLOUT_LEN] + agent.INPUT_SHAPE)
  rewards = np.zeros([ROLLOUT_LEN])
  actions = np.zeros([ROLLOUT_LEN], dtype=np.int)
  dones = np.zeros([ROLLOUT_LEN], dtype=np.bool)
  value_ests = np.zeros([ROLLOUT_LEN])
  act_log_probs = np.zeros([ROLLOUT_LEN])

  episode_rewards = []

  for i in range(ROLLOUT_LEN):
    states[i] = state
    if not agent.ENVIRONMENT == 'CartPole-v1':
      state = state._force()
    policy_logits, value_est = actor.policy_and_value(np.expand_dims(state, 0))
    action, act_log_prob = actor.act(policy_logits)
    state,reward,done,info = env.step(action[0].numpy())
    rewards[i] = reward
    actions[i] = action
    dones[i] = done
    value_ests[i] = value_est
    act_log_probs[i] = act_log_prob
    cumulative_reward += reward
    if done:     
      state = env.reset()
      episode_rewards += [cumulative_reward]
      cumulative_reward = 0

  advs = np.zeros((ROLLOUT_LEN,))

  # implementation follows https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L56-L65
  for t in reversed(range(ROLLOUT_LEN)):
    if t == ROLLOUT_LEN - 1:
      nextnonterminal = 1.0 - done
      nextvalue = value_est
      next_adv = 0.0
      #nextreturn = value_est
    else:
      nextnonterminal = 1.0 - dones[t+1]
      nextvalue = value_ests[t+1]
      #nextreturn = returns[t+1]
    delta = rewards[t] + agent.DISCOUNT * nextvalue * nextnonterminal - value_ests[t]
    next_adv = delta + agent.LAMBDA * agent.DISCOUNT * next_adv * nextnonterminal 
    advs[t] = next_adv
    #returns[t] = rewards[t] + agent.DISCOUNT * nextreturn * nextnonterminal
  returns = advs + value_ests

    
  return (states, actions, dones, returns, value_ests, act_log_probs, state, cumulative_reward, episode_rewards)
  


if __name__ == '__main__':

  # need to spawn processes so they don't copy tensorflow state and try to do stuff on GPU
  #mp.set_start_method('spawn')

  loop_i = 0
  if loop_i == 2 and PROFILE:
    tf.profiler.experimental.start('logdir')
  if loop_i == 4 and PROFILE:
    tf.profiler.experimental.stop()
  loop_i += 1;
  

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
  
  envs = []
  states = []
  cumulative_rewards = []

  # make environment
  for i in range(ENVS):
    env = makeEnv()
    state = env.reset()
    envs += [env]
    cumulative_rewards += [0]
    states.append(state)
  
  cycle = 0


  while True: 
    if cycle * FRAMES_PER_CYCLE > agent.FRAMES:
      break;


    states_l, actions_l, dones_l, returns_l, value_ests_l, log_probs_l, next_states, next_cumulative_rewards = [],[],[],[],[],[],[],[]
    data_lists = [states_l, actions_l, dones_l, returns_l, value_ests_l, log_probs_l, next_states, next_cumulative_rewards]
    for (e,s,cr) in zip(envs, states, cumulative_rewards):
      dataBatch = actorProcess(e,s,cr,actor)
      for i,dl in enumerate(data_lists):
        dl.append(dataBatch[i])
      agentsave['episode_rewards'] += dataBatch[-1]

    states, cumulative_rewards = (next_states, next_cumulative_rewards)

    data_arrays = []
    for dl in data_lists[:-2]:
      data_arrays.append(np.concatenate(dl))

    if len(agentsave['episode_rewards']) > 0:
      av_ep_rew = sum(agentsave['episode_rewards'][-100:]) / len(agentsave['episode_rewards'][-100:])
      print('Average episode reward: ' + str(av_ep_rew))
    

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
    for e in range(EPOCHS):
      #print('epoch ' + str(e))
      np.random.shuffle(indices)
      for mb_start in range(0,total_datapoints,MINIBATCH_SIZE):
        mb_indices = indices[mb_start:mb_start + MINIBATCH_SIZE]
        inputs = [d[mb_indices,...] for d in data_arrays]
        mb_states, mb_actions, mb_dones, mb_returns, mb_old_values, mb_log_action_probs = inputs 
  
        loss_pve = actor.train(mb_states, mb_actions, mb_returns, mb_old_values, mb_log_action_probs)
        #print("{:6f}, {:6f}, {:6f}".format(loss_pve[0].numpy(), loss_pve[1].numpy(), loss_pve[2].numpy()))

        loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss_pve)
  
        agentsave['loss_policy'] += [loss_pve[0].numpy()]
        agentsave['loss_value'] += [loss_pve[1].numpy()]
        agentsave['loss_entropy'] += [loss_pve[2].numpy()]
        
    print(loss_str)
       
      
  
  
    cycle += 1
    if not cycle % SAVE_CYCLES:
    #if True: # save every time, because currently we get model by loading from disk on each actor process
      print('Saving model...')
      tf.saved_model.save(actor, agent.model_savepath)
      with open(agent.picklepath, "wb") as fp:
        pickle.dump(agentsave, fp)
      
  
    
  
  
  
