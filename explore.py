import tensorflow as tf
import argparse
from datetime import datetime
import code
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
from replay import ReplayBuffer
from running_mean_std import RunningMeanStd


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent
import wgan
from vecenv import VecEnv # not parallelized yet

from atari_wrappers import WarpFrame, NoopResetEnv, EpisodicLifeEnv, ClipRewardEnv, FrameStack, ScaledFloatFrame, MaxAndSkipEnv, FireResetEnv

# makes replay buffer smaller so it doesn't take so long to fill it.
# turn off when not debugging.
QUICKTEST = False
TEST_ONLY_WGAN = True

INIT_BUFFER_FILE = 'random_policy_replay_buffer.npy'


def makeEnv():
  env = gym.make(agent.ENVIRONMENT)
  if not agent.ENVIRONMENT == 'CartPole-v1' and not agent.ENVIRONMENT == 'Acrobot-v1':
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    #env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
      env = FireResetEnv(env)
    env = WarpFrame(env, 84, 84)
    env = ScaledFloatFrame(env) # negates optimization of LazyFrames in FrameStack
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.TimeLimit(env, 18000) # exploration by RND paper uses this
  return env

# only run PPO training if wgan critic loss is higher than this threshold
CRITIC_THRESHOLD_FOR_PPO = -100000
# testing with only wgan training rn
INITIAL_WGAN_TRAINING_CYCLES = 1000000

# 2**18 is the largest my ram can handle currently
# seems too low, TODO optimize
WGAN_BUFFER_SIZE = 2**20
if QUICKTEST:
  WGAN_BUFFER_SIZE = 2**14

INTRINSIC_REWARD_MULTIPLIER = agent.INTRINSIC_REWARD_MULTIPLIER

PROFILE = True

TRAIN_PPO = True
SAVE_CYCLES = 20
ENVS = agent.ENVS
ROLLOUT_LEN = agent.ROLLOUT_LEN
MINIBATCH_SIZE = agent.MINIBATCH_SIZE
MINIBATCHES = ENVS * ROLLOUT_LEN // MINIBATCH_SIZE # must be multiple of 
FRAMES_PER_CYCLE = ENVS * ROLLOUT_LEN
EPOCHS = agent.EPOCHS
WGAN_BATCH_SIZE = 64
WGAN_TRAINING_CYCLES = 1
CRITIC_BATCHES = 20
GEN_BATCHES = 1
PARAM_UPDATES_PER_CYCLE = MINIBATCHES * EPOCHS

NP_FLOAT_TYPE = np.float16


''' 
given vecenv and actor, and RunningMeanStd object,
computes the generalized advantage estimate for a rollout of n steps.

returns:
[states, actions, dones, returns value_ests, act_log_probs], episode_rewards

episode_rewards is a list of total reward for any episodes that terminated during
the rollout.
'''
def getRollout(vecenv, n, actor, get_intrinsic_reward=None, rms=None):

  # + 1 to hold extra state after last transition, to use to get final one-step td estimate
  states = np.zeros([n + 1, vecenv.nenvs] + agent.INPUT_SHAPE, dtype=np.float32)
  extrinsic_rewards = np.zeros([n, vecenv.nenvs])
  actions = np.zeros([n, vecenv.nenvs], dtype=np.int)
  dones = np.zeros([n, vecenv.nenvs], dtype=np.bool)

  act_probs = np.zeros([n,vecenv.nenvs])


  episode_scores = []

  states[0] = vecenv.getStates()

  for i in range(n):
    policy_logits = actor.policy(vecenv.getStates())
    actions[i], act_probs[i] = actor.act(policy_logits)
    states[i+1], extrinsic_rewards[i], dones[i], ep_scores = vecenv.step(actions[i])
    episode_scores += ep_scores

  if get_intrinsic_reward == None:
    return [states], episode_scores

  # flatten batch dim and env dim, to comute intrinsic rewards in one batch
  reshaped_states = np.reshape(states[:-1], (n*vecenv.nenvs,) + states.shape[2:])
  intrinsic_rewards = get_intrinsic_reward(reshaped_states[..., -1:])

  rms.update(intrinsic_rewards)
  intrinsic_rewards = intrinsic_rewards - rms.mean
  intrinsic_rewards = intrinsic_rewards / (rms.var ** 0.5)
  print('var: ' + str(rms.var))
  print('intrinsic mean: ' + str(np.mean(intrinsic_rewards)))

  intrinsic_rewards = np.reshape(intrinsic_rewards, (n, vecenv.nenvs) + intrinsic_rewards.shape[1:])
  

  return (states, extrinsic_rewards, intrinsic_rewards, actions, dones, act_probs), episode_scores

''' given rollout = (states, rewards, actions, dones, act_log_probs)
as returned by getRollout, n = length of rollout, and actor, 
returns the GAE adv estimates and returns,
as well as the rest of the inputs needed for the batch of training

TODO speed this up, it's too slow (not worth recomputing every epoch)
'''
def getGAE(rollout, actor):
  n = rollout[0].shape[0] - 1
  numstates = n + 1
  nenvs = rollout[0].shape[1]

  (states, extrinsic_rewards, intrinsic_rewards, actions, dones, act_probs) = rollout
  reshaped_states = np.reshape(states, (numstates*nenvs,) + states.shape[2:])
  extrinsic_vals, intrinsic_vals = actor.value(reshaped_states)

  extrinsic_vals = np.reshape(extrinsic_vals, (numstates, nenvs) + extrinsic_vals.shape[1:])
  intrinsic_vals = np.reshape(intrinsic_vals, (numstates, nenvs) + intrinsic_vals.shape[1:])

  intrinsic_advs = np.zeros((n,vecenv.nenvs), NP_FLOAT_TYPE)
  extrinsic_advs = np.zeros((n,vecenv.nenvs), NP_FLOAT_TYPE)

  intrinsic_next_adv = 0.0
  extrinsic_next_adv = 0.0

  # compute returns and advs using Generalized Advantage Estimation
  for t in reversed(range(n)):
    nonterminal = (1 - dones[t])

    # note: ignore episode terminal for intrinsic reward calculation
    # don't want to discourage the agent from dieing so that it becomes too cautious and doesn't explore
    # frontier states.
    extrinsic_delta = extrinsic_rewards[t] + agent.EXTRINSIC_DISCOUNT * extrinsic_vals[t+1] * nonterminal - extrinsic_vals[t]
    extrinsic_advs[t] = extrinsic_delta + agent.LAMBDA * agent.EXTRINSIC_DISCOUNT * extrinsic_next_adv * nonterminal 
    extrinsic_next_adv = extrinsic_advs[t]

    intrinsic_delta = intrinsic_rewards[t] + agent.INTRINSIC_DISCOUNT * intrinsic_vals[t+1] - intrinsic_vals[t]
    intrinsic_advs[t] = intrinsic_delta + agent.LAMBDA * agent.INTRINSIC_DISCOUNT * intrinsic_next_adv
    intrinsic_next_adv = intrinsic_advs[t]

  states = states[:-1,...]          # toss last state and value estimage

  val_ests = extrinsic_vals + agent.INTRINSIC_REWARD_MULTIPLIER * intrinsic_vals 
  val_ests = val_ests[:-1,...]

  extrinsic_returns = extrinsic_advs + extrinsic_vals[:-1]
  intrinsic_returns = intrinsic_advs + intrinsic_vals[:-1]

  # reshape all the collected data into a single batch before returning
  batch = [states, actions, extrinsic_returns, intrinsic_returns, val_ests, act_probs]
  for i,x in enumerate(batch):
    batch[i] = np.reshape(x, (n*vecenv.nenvs,) + x.shape[2:])

  return batch

def updateWganReplay(rollout, replay_buffer):
  # throw out last state, or we'd be putting it in twice (getRollout returns
  # one more state than the length of a rollout)
  states = rollout[0][:-1] 
  states = np.reshape(states, (states.shape[0] * states.shape[1],) + states.shape[2:])
  states = states.astype(NP_FLOAT_TYPE)
  # only add one state frame to buffer
  replay_buffer.add(states[...,-1:]) 


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="explore with |wgan critic loss| intrinsic reward")
  parser.add_argument('--load_model', dest="load_model", default='', type=str, help="path to model to load")
  parser.add_argument('--savepath', dest="savepath", default='', type=str, help="path where model will be saved")
  args = parser.parse_args()

  if args.savepath:
    savepath = args.savepath
  elif args.load_model:
    savepath = args.load_model
  else:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    savedir_path = os.path.join(dir_path, 'saves')
    savepath = os.path.join(savedir_path, 'explore_' + str(datetime.now()))
  wgan_savepath = os.path.join(savepath, 'wgan')
  actor_savepath = os.path.join(savepath, 'actor')
  picklepath = os.path.join(savepath, 'save.pickle')
  
  # load model if specified, otherwise create model
  if (args.load_model):
    actor = tf.saved_model.load(actor_savepath)
    wgan = tf.saved_model.load(wgan_savepath)
  else:
    gan = wgan.WGAN();
    actor = agent.Agent();
    tf.saved_model.save(actor, actor_savepath)
    tf.saved_model.save(gan, wgan_savepath)
    save = {}
    with open(picklepath, "wb") as fp:
      pickle.dump(save, fp)

  with open(picklepath, "rb") as f: 
    save = pickle.load(f)
    defaults = {
      'loss_policy': [], 
      'loss_value': [], 
      'loss_entropy': [],
      'episode_rewards': [],
      'gen_losses': [],
      'critic_losses': [],
      'framecount': 0,
      'ppo_iters': 0,
      'wgan_cycles': 0,
    }
    for s,v in defaults.items():
      if not s in save:
        save[s] = v

  def getIntrinsicReward(states):
    # TODO do we want the first or last stacked frame?
    scores = gan.critic(states[...,-1:]) # last index is time index.
    return tf.abs(scores) * INTRINSIC_REWARD_MULTIPLIER

  vecenv = VecEnv(makeEnv, ENVS)
  rms = RunningMeanStd()
  

  wgan_in_shape = wgan.IMSPEC.shape.as_list()[1:]
  wgan_replay_buffer = ReplayBuffer(WGAN_BUFFER_SIZE, wgan_in_shape, NP_FLOAT_TYPE)

  def datasetGenerator():
    while True:
      data = wgan_replay_buffer.sample(WGAN_BATCH_SIZE)
      yield data

  output_sig = tf.TensorSpec([WGAN_BATCH_SIZE] + wgan_in_shape, dtype=wgan.FLOAT_TYPE)
  dataset = tf.data.Dataset.from_generator(datasetGenerator, output_signature=output_sig)
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
  dataiter = iter(dataset)

  # so we don't have to randomly act to fill up the replay buffer every time 
  # we re-run this with different hyperparams
  if os.path.exists(INIT_BUFFER_FILE):
    wgan_replay_buffer.loadFromFile(INIT_BUFFER_FILE)
  else:

    # fill up replay buffer before starting
    while not wgan_replay_buffer.isFull():
      save['framecount'] += ROLLOUT_LEN * vecenv.nenvs
      rollout, episode_rewards = getRollout(vecenv, ROLLOUT_LEN, actor) 
      #[states, rewards, actions, dones, act_log_probs] = rollout
      updateWganReplay(rollout, wgan_replay_buffer)
    wgan_replay_buffer.saveToFile(INIT_BUFFER_FILE)

  cycle = -1
  while True:
  #for cycle in range(0, agent.FRAMES // FRAMES_PER_CYCLE):
    cycle += 1

    if cycle == 2 and PROFILE:
      tf.profiler.experimental.start('logdir')
    if cycle == 3 and PROFILE:
      tf.profiler.experimental.stop()


    if len(save['episode_rewards']) > 0:
      av_ep_rew = sum(save['episode_rewards'][-20:]) / len(save['episode_rewards'][-20:])
      print('Average episode reward: ' + str(av_ep_rew))
      print('Episode count: ' + str(len(save['episode_rewards'])))

    print("Frames: %d" % save['framecount'])

    if TEST_ONLY_WGAN:
      rollout, episode_rewards = getRollout(vecenv, 8, actor, getIntrinsicReward, rms)
      updateWganReplay(rollout, wgan_replay_buffer)

    print('training WGAN')
    for c in range(WGAN_TRAINING_CYCLES):
      

      save['wgan_cycles'] += 1
      for _ in range(CRITIC_BATCHES):
        real_images = next(dataiter)

        critic_loss = gan.trainCritic(real_images, WGAN_BATCH_SIZE)
        save['critic_losses'] += [critic_loss]
      for _ in range(GEN_BATCHES):
        gen_loss = gan.trainGen(WGAN_BATCH_SIZE)
        save['gen_losses'] += [gen_loss]
      loss_str = ''.join('{:6f}, '.format(lossv) for lossv in (critic_loss, gen_loss))
      print('critic, gen: ' + loss_str)
       
  
    if TRAIN_PPO and critic_loss > CRITIC_THRESHOLD_FOR_PPO and cycle > INITIAL_WGAN_TRAINING_CYCLES:
      save['framecount'] += ROLLOUT_LEN * vecenv.nenvs
      save['ppo_iters'] += 1
      rollout, episode_rewards = getRollout(vecenv, ROLLOUT_LEN, actor, getIntrinsicReward, rms)
      #[states, rewards, actions, dones, act_log_probs] = rollout
      updateWganReplay(rollout, wgan_replay_buffer)

      save['episode_rewards'] += episode_rewards

      print('training ppo')
      total_datapoints = MINIBATCH_SIZE * MINIBATCHES
      indices = np.arange(total_datapoints)
      # TODO: does it help sample complexity to recompute GAE at each epoch? some implementations do this (minimalRL), some dont (stable baselines 3)
      # this paper https://arxiv.org/pdf/2006.05990.pdf recommends recomputing them for each epoch
      # but, my current implementation of getGAE is really slow
      batch = getGAE(rollout, actor)
      for e in range(EPOCHS):

        np.random.shuffle(indices)
        for mb_start in range(0,total_datapoints,MINIBATCH_SIZE):
          mb_indices = indices[mb_start:mb_start + MINIBATCH_SIZE]
          inputs = [d[mb_indices,...] for d in batch]
  
          loss_pve, grads, r, adv = actor.train(*inputs)

          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss_pve)
  
          save['loss_policy'] += [loss_pve[0].numpy()]
          save['loss_value'] += [loss_pve[1].numpy()]
          save['loss_entropy'] += [loss_pve[2].numpy()]
          
      print(loss_str)
  
    if not cycle % SAVE_CYCLES:
      print('Saving model...')
      tf.saved_model.save(actor, actor_savepath)
      tf.saved_model.save(gan, wgan_savepath)
      with open(picklepath, "wb") as fp:
        pickle.dump(save, fp)
  
    
  
  
  
