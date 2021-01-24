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
ENVIRONMENT = 'Pong-v0'

env = gym.make(ENVIRONMENT)
ACTIONS = env.action_space.n

WIDTH = 84
HEIGHT = 110
DEPTH = 4

FEATURE_SIZE = 32
HIDDEN_NEURONS=128

FILTER_SIZES = [3, 5, 5, 7]
CHANNELS =     [8,16,32,32]
STRIDES =     [2,2,2,1]

IMSPEC = tf.TensorSpec([None, WIDTH, HEIGHT, DEPTH])
INTSPEC = tf.TensorSpec([None], dtype=tf.int64)
FLOATSPEC = tf.TensorSpec([None],)
BOOLSPEC = tf.TensorSpec([None], dtype=tf.bool)
LOGITSPEC = tf.TensorSpec([None, ACTIONS])

DISCOUNT = 1
ENTROPY_WEIGHT = 0.01
EPSILON = 0.1

# TODO the entropy scale is way off. Probably computing it wrong...
ADD_ENTROPY = True

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'actor.mod')
loss_savepath = os.path.join(dir_path, 'actor_loss.pickle')

def getConvOutputSize(w,h,filtersize, channels, stride):
  # padding if necessary
  w = math.ceil(w / stride)
  h = math.ceil(h / stride)
  return w,h,channels

  
  


class Agent(Model):
  def __init__(self):
    super(Agent, self).__init__()
    self.vars = []
    self.critic_vars = []
    self.gen_vars = []

    mvars = []
    size = (WIDTH,HEIGHT,DEPTH)
    for (f, c, s) in zip(FILTER_SIZES, CHANNELS, STRIDES):
      # first conv layer
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='agent_conv'))
      size = getConvOutputSize(size[0], size[1], f, c, s)


    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(size[0]*size[1]*size[2],HIDDEN_NEURONS)), name='Value_w'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='Value_b'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,1)), name='Value_o'))

    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(size[0]*size[1]*size[2],HIDDEN_NEURONS)), name='Policy_w'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='Policy_b'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ACTIONS)), name='Policy_o'))

    self.vars += mvars


  ''' output policy is the policy logits, softmax must be taken later. '''
  @tf.function(input_signature=(IMSPEC,))
  def policy_and_value(self, states):
    mvars = self.vars
    x = states
    for i in range(len(CHANNELS)):
      filt = mvars[i]
      stride = STRIDES[i]
      x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
      x = tf.nn.leaky_relu(x)


    vi = len(CHANNELS)
    w, b = mvars[vi:vi+2]
    x = tf.keras.layers.Flatten()(x)

    value= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w = mvars[vi+2]
    value = tf.squeeze(tf.einsum('ba,ao->bo',value,w))

    w, b = mvars[vi+3:vi+5]
    policy= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w = mvars[vi+5]
    policy = tf.einsum('ba,ao->bo',policy,w)
  
    return policy, value

  ''' returns the action and the probability of having taken that action '''
  @tf.function(input_signature=(LOGITSPEC,))
  def act(self, logits):
    actions = tf.squeeze(tf.random.categorical(logits, 1))
    action_probs = tf.nn.softmax(logits)
    action_probs = tf.squeeze(tf.gather(action_probs, actions, batch_dims=1))
    return actions, action_probs


  @tf.function(input_signature=(IMSPEC, INTSPEC, FLOATSPEC, FLOATSPEC))
  def loss(self, states, actions, old_action_probs, value_targets):
    policy_logits, value_est = self.policy_and_value(states)
    advantage = value_targets - value_est
    #print('value_est')
    #print(value_est)
    value_loss = tf.reduce_mean(tf.pow(advantage, 2))
    #print('value_loss')
    #print(value_loss)
    # for the policy loss, stop gradients on advantage
    advantage = tf.stop_gradient(advantage)

    policy_probs = tf.nn.softmax(policy_logits)
    #print('policy_probs:')
    #print(policy_probs)
    action_probs = tf.squeeze(tf.gather(policy_probs, actions, batch_dims=1))
    r = action_probs / old_action_probs
    #print('r')
    #print(r)
    
    # select the policy action probabilities of the given actions
    #print('actions, policy_probs, action_probs')
    #print(actions)
    #print(policy_probs)
    #print(action_probs)
    #print(tf.math.logical_and(tf.math.greater(r, 1-EPSILON), tf.math.less(r, 1+EPSILON)))
    policy_loss = -1.0 * tf.reduce_mean(tf.math.minimum(r * advantage, tf.clip_by_value(r, 1-EPSILON, 1+EPSILON) * advantage))

    # compute the entropy of the policy for entropy loss
    exps = tf.exp(policy_logits)
    log_denom = tf.math.log(tf.reduce_sum(exps, axis=-1, keepdims=True))
    entropy_loss = -1.0 * ENTROPY_WEIGHT * tf.reduce_mean(policy_probs * (policy_logits - log_denom))

    loss = policy_loss + value_loss
    if ADD_ENTROPY:
      loss += ENTROPY_WEIGHT * entropy_loss

    # TODO: remove. Just to test critic
    #policy_loss = tf.zeros((0,))
    entropy_loss = tf.zeros((0,))

    return policy_loss, value_loss, entropy_loss
    

    
    
if __name__ == '__main__':
  agent = Agent();

  #print('running loss')
  #encoder.loss_from_beginning(tf.zeros((16,84,110,1)))

  
  print('Saving model...')
  tf.saved_model.save(agent, model_savepath)


  losses = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
    

    
    




    
