# TODO add optimizers to this model, so they can be saved and loaded easily

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import math
import os
from tensorflow.keras import Model
import gym
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# For some reason this is necessary to prevent error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


ENVIRONMENT = 'MontezumaRevenge-v0'
#ENVIRONMENT = 'PongDeterministic-v4'
#ENVIRONMENT = 'CartPole-v1'

env = gym.make(ENVIRONMENT)
ACTIONS = env.action_space.n

WIDTH = 84
HEIGHT = 110
DEPTH = 4


ENT_EPSILON = 1e-7

HIDDEN_NEURONS=128

FILTER_SIZES = [8, 4, 3]
#FILTER_SIZES = []
CHANNELS =     [32,64,64]
#CHANNELS = []
STRIDES =     [4,2,1]
#STRIDES = []


IMSPEC = tf.TensorSpec([None, WIDTH, HEIGHT, DEPTH])
if ENVIRONMENT == 'CartPole-v1':
  IMSPEC = tf.TensorSpec([None, 4])

INTSPEC = tf.TensorSpec([None], dtype=tf.int64)
FLOATSPEC = tf.TensorSpec([None],)
BOOLSPEC = tf.TensorSpec([None], dtype=tf.bool)
LOGITSPEC = tf.TensorSpec([None, ACTIONS])

DISCOUNT = 0.99
ENTROPY_WEIGHT = 0.01
EPSILON = 0.1

#ADD_ENTROPY = True

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'actor_save')
loss_savepath = os.path.join(dir_path, 'actor_loss.pickle')
rewards_savepath = os.path.join(dir_path, 'rewards.pickle')



def getConvOutputSizeValid(w,h,filtersize, channels, stride):
  # padding if necessary
  w = (w - filtersize) // stride + 1
  h = (h - filtersize) // stride + 1
  return w,h,channels

  
  


class Agent(tf.Module):
  def __init__(self):
    super(Agent, self).__init__()
    self.vars = []

    size = (WIDTH,HEIGHT,DEPTH)
    if ENVIRONMENT == 'CartPole-v1':
      size = (4,1,1)


    for (f, c, s) in zip(FILTER_SIZES, CHANNELS, STRIDES):
      # first conv layer
      self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='agent_conv'))
      size = getConvOutputSizeValid(size[0], size[1], f, c, s)


    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(size[0]*size[1]*size[2],HIDDEN_NEURONS)), name='Value_w'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='Value_b'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,1)), name='Value_o'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(1,)), name='Value_bo'))

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(size[0]*size[1]*size[2],HIDDEN_NEURONS)), name='Policy_w'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='Policy_b'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ACTIONS)), name='Policy_o'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ACTIONS,)), name='Policy_bo'))


  ''' output policy is the policy logits, softmax must be taken later. '''
  @tf.function(input_signature=(IMSPEC,))
  def policy_and_value(self, states):
    #print("REMAKING GRAPH FOR POLICY_AND_VALUE")
    mvars = self.vars
    x = states
    for i in range(len(CHANNELS)):
      filt = mvars[i]
      stride = STRIDES[i]
      x = tf.nn.conv2d(x,filt,stride,'VALID',name=None)
      x = tf.nn.leaky_relu(x)

    x = tf.keras.layers.Flatten()(x)

    vi = len(CHANNELS)
    w, b = mvars[vi:vi+2]
    value= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w,b = mvars[vi+2:vi+4]
    value = tf.squeeze(tf.einsum('ba,ao->bo',value,w))  + b

    vi += 4
    w, b = mvars[vi:vi+2]
    policy= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w, b = mvars[vi+2:vi+4]
    policy = tf.squeeze(tf.einsum('ba,ao->bo',policy,w))  + b

    # TODO does this actually work?
    #policy = tf.abs(policy)

  
    return policy, value

  ''' returns the action and the probability of having taken that action '''
  @tf.function(input_signature=(LOGITSPEC,))
  def act(self, logits):
    actions = tf.squeeze(tf.random.categorical(logits, 1))
    action_probs = tf.nn.softmax(logits, axis=-1)
    action_probs = tf.squeeze(tf.gather(action_probs, actions, batch_dims=1))
    return actions, action_probs


  @tf.function(input_signature=(IMSPEC, INTSPEC, FLOATSPEC, FLOATSPEC))
  def loss(self, states, actions, old_action_probs, value_targets):
    policy_logits, value_est = self.policy_and_value(states)
    advantage = value_targets - value_est
    value_loss = tf.reduce_mean(tf.pow(advantage, 2))
    # for the policy loss, stop gradients on advantage
    advantage = tf.stop_gradient(advantage)

    policy_probs = tf.nn.softmax(policy_logits, axis=-1)
    action_probs = tf.squeeze(tf.gather(policy_probs, actions, batch_dims=1))
    r = action_probs / old_action_probs
    
    # select the policy action probabilities of the given actions
    policy_loss = -1.0 * tf.reduce_mean(tf.math.minimum(r * advantage, tf.clip_by_value(r, 1-EPSILON, 1+EPSILON) * advantage))

    # compute the entropy of the policy for entropy loss
    # minimize the log instead of entropy directly so that gradient signals are stronger at deterministic policies
    #entropy_loss =  -1.0 * ENTROPY_WEIGHT * tf.reduce_mean(tf.math.log(probs_logits_entropy(policy_probs, policy_logits)))
    entropy_loss =  -1.0 * ENTROPY_WEIGHT * tf.reduce_mean((probs_logits_entropy(policy_probs, policy_logits)))

    #tf.print(policy_probs)

    #return (entropy_loss,)
    return policy_loss, value_loss, entropy_loss
    

# OKAAAAAY SO. when I use any  attempt to actually calculate the gradients,
# the network quickly reaches a deterministic state and then the gradients are all 0 so
# it never leaves it. If the network structure is setup so it doesn't reach a deterministic
# state easily, then this can work, but I don't like the idea of losing all gradients in 
# deterministic states. SO I decided to just hack in a loss that penalizes logits 
# differing from their mean value. It seems to work fine on cartpole...
def probs_logits_entropy(probs, logits):

  probs = probs + ENT_EPSILON # for numerical stability at deterministic policies
  #tf.print(logits)
  #logit_mean = tf.reduce_mean(logits, axis=-1, keepdims=True)
  #return -1.0 * tf.reduce_sum(tf.abs((logits - logit_mean) / (logit_mean)), axis=-1)
  #return -1.0 * tf.reduce_sum(probs * tf.math.log(probs + 1e-7))

  ent =  tf.nn.softmax_cross_entropy_with_logits(probs, logits, axis=-1)
  return ent

    
    
if __name__ == '__main__':

  #print('running loss')
  #encoder.loss_from_beginning(tf.zeros((16,84,110,1)))

  

  agent = Agent();

  print('Saving model...')
  tf.saved_model.save(agent, model_savepath)


  losses = []
  episode_rewards = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
  with open(rewards_savepath, "wb") as fp:
    pickle.dump(episode_rewards, fp)
    

    
    




    
