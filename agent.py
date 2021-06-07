# TODO add optimizers to this model, so they can be saved and loaded easily

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import math
from datetime import datetime
import os
from tensorflow.keras import Model
import gym
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

DEBUG = True

#TODO move hyperparameters into constructor arguments

# NOTE: need to use NoFrameskip versions of environments, since we are wraping
# env with MaxAndSkipEnv wrapper in agent.
ENVIRONMENT = 'MontezumaRevengeNoFrameskip-v4'
#ENVIRONMENT = 'PongNoFrameskip-v4'
#ENVIRONMENT = 'CartPole-v1'
#ENVIRONMENT = 'Acrobot-v1'

SQRT2 = 2.0**0.5

env = gym.make(ENVIRONMENT)
ACTIONS = env.action_space.n

INTRINSIC_REWARD_MULTIPLIER = 0.01

CLIPNORM = 0.5
# a higher epsilon prevents nans in gradient applications when parameters have very low variance
ADAM_PARAMS = {'global_clipnorm': CLIPNORM, 'epsilon': 1e-5} 

FLOAT_TYPE = tf.float32
INT_TYPE = tf.int32

if ENVIRONMENT == 'CartPole-v1':
  WIDTH,HEIGHT,DEPTH= (4,1,1)
  ACTIVATION = tf.math.tanh
  #ACTIVATION = tf.nn.leaky_relu
  #ACTIVATION = tf.nn.relu
  VALUE_LOSS_WEIGHT = 0.5 # stable-baselines-3 uses 0.5 by default. 
  ENVS = 1
  STRIDES = []
  CHANNELS = []
  FILTER_SIZES = []
  INPUT_SHAPE = [4]
  FRAMES = 100000
  MINIBATCH_SIZE = 64
  VAL_LAYERS = [64,64]
  POL_LAYERS = [64,64]
  IMSPEC = tf.TensorSpec([None, 4], dtype=FLOAT_TYPE)
  ENTROPY_WEIGHT = 0.00
  EPSILON = 0.2
  EPOCHS = 10
  ROLLOUT_LEN = 2048
  START_LR = 0.0003
  END_LR = 0.0003

elif ENVIRONMENT=='Acrobot-v1':
  WIDTH,HEIGHT,DEPTH= (6,1,1)
  ACTIVATION = tf.math.tanh
  VALUE_LOSS_WEIGHT = 0.5 # stable-baselines-3 uses 0.5 by default. 
  ENVS = 1
  VAL_LAYERS = [64,64]
  POL_LAYERS = [64,64]
  STRIDES = []
  CHANNELS = []
  FILTER_SIZES = []
  INPUT_SHAPE = [6]
  FRAMES = 1000000
  MINIBATCH_SIZE = 64
  IMSPEC = tf.TensorSpec([None, 6], dtype=FLOAT_TYPE)
  ENTROPY_WEIGHT = 0.00
  EPSILON = 0.2
  EPOCHS = 10
  ROLLOUT_LEN = 2048
  START_LR = 0.0003
  END_LR = 0.0003

else: # atari

  WIDTH,HEIGHT,DEPTH=(84,84,4)
  ACTIVATION = tf.nn.leaky_relu
  VALUE_LOSS_WEIGHT = 1 # PPO paper uses 1.0 for atari
  ENVS = 32
  FILTER_SIZES = [8, 4,3]
  CHANNELS =     [16,32,32]
  STRIDES =     [4,2,1]
  FRAMES = 40000000
  MINIBATCH_SIZE = 32 * 8
  VAL_LAYERS = [256]
  POL_LAYERS = [256]
  IMSPEC = tf.TensorSpec([None, WIDTH, HEIGHT, DEPTH], dtype=FLOAT_TYPE)
  ENTROPY_WEIGHT = 0.01 
  EPSILON = 0.1
  EPOCHS = 4
  ROLLOUT_LEN = 128
  START_LR = 0.00025
  END_LR = 0.0
  INPUT_SHAPE = [WIDTH, HEIGHT, DEPTH]
  PADDING = 'SAME'


INTRINSIC_DISCOUNT = 0.99
EXTRINSIC_DISCOUNT = 0.999
LAMBDA = 0.95 # for generalized advantage estimate


ENT_EPSILON = 1e-7


INTSPEC = tf.TensorSpec([None], dtype=INT_TYPE)
FLOATSPEC = tf.TensorSpec([None],dtype=FLOAT_TYPE)
BOOLSPEC = tf.TensorSpec([None], dtype=tf.bool)
LOGITSPEC = tf.TensorSpec([None, ACTIONS],dtype=FLOAT_TYPE)


dir_path = os.path.dirname(os.path.realpath(__file__))
savedir_path = os.path.join(dir_path, 'saves')
model_savepath = os.path.join(savedir_path, 'ppo_save')
backup_savepath = os.path.join(savedir_path, ENVIRONMENT + str(datetime.now()))
picklepath = os.path.join(model_savepath, 'ppo.pickle')
picklepath_backup = os.path.join(backup_savepath, 'ppo.pickle')


def getConvOutputSize(w,h,filtersize, channels, stride, padding='SAME'):
  if padding == 'VALID':
    w = (w - filtersize) // stride + 1
    h = (h - filtersize) // stride + 1
    return w,h,channels
  if padding == 'SAME':
    w = math.ceil(w / stride)
    h = math.ceil(h / stride)
    return w,h,channels


class Agent(tf.Module):
  def __init__(self):
    super(Agent, self).__init__()
    self.vars = []
    self.valvars = []
    self.polvars = []

    size = INPUT_SHAPE
    for (f, c, s) in zip(FILTER_SIZES, CHANNELS, STRIDES):
      # first conv layer
      self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c),dtype=FLOAT_TYPE), name='agent_conv',dtype=FLOAT_TYPE))
      size = getConvOutputSize(size[0], size[1], f, c, s, padding='SAME')

    startsize = np.prod(size)
    size = startsize
    for i,h in enumerate(VAL_LAYERS + [2]): # 2 value head streams for intrinsic v extrinsic rewards
      gain = SQRT2 if i < len(VAL_LAYERS) else 1.0
      self.valvars.append(tf.Variable(tf.initializers.Orthogonal(gain=gain)(shape=(size,h),dtype=FLOAT_TYPE), name='Value_w' + str(i), dtype=FLOAT_TYPE))
      self.valvars.append(tf.Variable(tf.zeros_initializer()(shape=(h,),dtype=FLOAT_TYPE), name='Value_b' + str(i),dtype=FLOAT_TYPE))
      size = h
    size = startsize
    for i,h in enumerate(POL_LAYERS + [ACTIONS]):
      gain = SQRT2 if i < len(POL_LAYERS) else 0.01
      self.polvars.append(tf.Variable(tf.initializers.Orthogonal(gain=gain)(shape=(size,h),dtype=FLOAT_TYPE), name='Policy_w' + str(i),dtype=FLOAT_TYPE))
      self.polvars.append(tf.Variable(tf.zeros_initializer()(shape=(h,),dtype=FLOAT_TYPE), name='Policy_b' + str(i),dtype=FLOAT_TYPE))
      size = h

    self.lr = tf.keras.optimizers.schedules.PolynomialDecay(
      START_LR,                       # start learning rate
      FRAMES // MINIBATCH_SIZE * EPOCHS,     # number of steps
      END_LR,                        # end learning rate
      power=1)
      
    ADAM_PARAMS['learning_rate'] = self.lr

    opt = tf.keras.optimizers.Adam
    self.opt = opt(**ADAM_PARAMS)
    #self.valopt = opt(**ADAM_PARAMS)
    #self.polopt = opt(**ADAM_PARAMS)
    self.vars += self.valvars + self.polvars

  @tf.function(input_signature=(IMSPEC,))
  def shared_net(self, states):
    mvars = self.vars
    x = states
    for i in range(len(CHANNELS)):
      filt = mvars[i]
      stride = STRIDES[i]
      x = tf.nn.conv2d(x,filt,stride,PADDING,name=None)
      x = tf.nn.leaky_relu(x)

    x = tf.keras.layers.Flatten()(x)
    return x

  @tf.function(input_signature=(IMSPEC,))
  def policy(self, states):
    policy = self.shared_net(states)
    for i,_ in enumerate(POL_LAYERS + [ACTIONS]):
      w, b = self.polvars[2*i:2*i + 2]
      policy = tf.einsum('ba,ah->bh', policy,w) + b
      if i < len(POL_LAYERS):
        policy = ACTIVATION(policy)
    return policy

  @tf.function(input_signature=(IMSPEC,))
  def value(self, states):
    value = self.shared_net(states)
    for i,_ in enumerate(VAL_LAYERS + [1]):
      w, b = self.valvars[2*i:2*i + 2]
      value = tf.einsum('ba,ah->bh', value,w) + b
      if i < len(VAL_LAYERS):
        value = ACTIVATION(value)
    extrinsic_val = value[:,0]
    intrinsic_val = value[:,1]
    return extrinsic_val, intrinsic_val


  # TODO will the below compute the shared net twice?
  ''' output policy is the policy logits '''
  @tf.function(input_signature=(IMSPEC,))
  def policy_and_value(self, states):
    policy = self.policy(states)
    extrinsic_val, intrinsic_val = self.value(states)
    return policy, extrinsic_val, intrinsic_val

  ''' returns the action and the probability of having taken that action '''
  @tf.function(input_signature=(LOGITSPEC,))
  def act(self, logits):
    actions = tf.random.categorical(logits, 1)
    probs = tf.nn.softmax(logits)
    action_probs = tf.gather(probs, actions, batch_dims=1)
    actions = tf.squeeze(actions, -1)
    action_probs = tf.squeeze(action_probs, -1)
    return actions, action_probs


  @tf.function(input_signature=(IMSPEC, INTSPEC, FLOATSPEC, FLOATSPEC, FLOATSPEC, FLOATSPEC))
  def train(self, states, actions, extrinsic_returns, intrinsic_returns, old_values, old_action_probs):
    advantage = (extrinsic_returns + intrinsic_returns - old_values)
    advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

    #value_est has dim [batch, 2] for the 2 value heads (extrinsic and intrinsic)
    policy_logits, intrinsic_val_est, extrinsic_val_est = self.policy_and_value(states)

    # learns significantly slower on cartpole with value fn clipping TODO
    # could this be an indiator to what is wrong? stable-baselines 3 also learns slower with 
    # value clipping, but not nearly as bad

    #clipped_value_est = old_values + tf.clip_by_value(value_est - old_values, -EPSILON, EPSILON)
    #vl1 = tf.square(value_est - returns)
    #vl2 = tf.square(clipped_value_est - returns)
    #value_loss = VALUE_LOSS_WEIGHT * tf.reduce_mean(tf.maximum(vl1, vl2))

    intrinsic_val_loss = tf.reduce_mean(tf.pow(intrinsic_val_est - intrinsic_returns,2))
    extrinsic_val_loss = tf.reduce_mean(tf.pow(extrinsic_val_est - extrinsic_returns,2))
    value_loss = VALUE_LOSS_WEIGHT * (intrinsic_val_loss + extrinsic_val_loss)

    probs = tf.nn.softmax(policy_logits)
    action_probs = tf.gather(probs, actions, batch_dims=1)

    r = action_probs / old_action_probs
    
    # select the policy action probabilities of the given actions
    mymin  = tf.math.minimum(r * advantage, tf.clip_by_value(r, 1.0-EPSILON, 1.0+EPSILON) * advantage)
    # Ignore objective if r*adv is out of bounds in the positive direction. (If negative, then we want to 
    # move policy back toward the old one)
    policy_loss = -1.0 * tf.reduce_mean(tf.math.minimum(r * advantage, tf.clip_by_value(r, 1.0-EPSILON, 1.0+EPSILON) * advantage))

    # compute the entropy of the policy for entropy loss
    policy_probs = tf.nn.softmax(policy_logits)
    entropy_loss =  -1.0 * ENTROPY_WEIGHT * tf.reduce_mean((probs_logits_entropy(policy_probs, policy_logits)))
   
    total_loss = policy_loss + value_loss + entropy_loss

    if DEBUG:
      for v in self.vars:
        tf.debugging.check_numerics(v, message="checking self.vars BEFORE update")

    #valgrads = tf.gradients(total_loss, self.valvars)
    #polgrads = tf.gradients(total_loss, self.polvars)
    #self.valopt.apply_gradients(zip(valgrads, self.valvars))
    #self.polopt.apply_gradients(zip(polgrads, self.polvars))
    grads = tf.gradients(total_loss, self.vars)
    self.opt.apply_gradients(zip(grads, self.vars))


    if DEBUG:
      tf.debugging.check_numerics(extrinsic_returns, message="checking ext ret")
      tf.debugging.check_numerics(intrinsic_returns, message="checking int ret")
      tf.debugging.check_numerics(old_values, message="checking old_values")

      tf.debugging.check_numerics(advantage, message="checking advantage")
      tf.debugging.check_numerics(policy_logits, message="checking policy_logits")
      tf.debugging.check_numerics(intrinsic_val_est, message="checking intrinsic value_est")
      tf.debugging.check_numerics(extrinsic_val_est, message="checking extrinsic value_est")
      tf.debugging.check_numerics(action_probs, message="checking action_probs")
      tf.debugging.check_numerics(r, message="checking r")
      tf.debugging.check_numerics(policy_probs, message="checking policy_probs")
      tf.debugging.check_numerics(entropy_loss, message="checking entropy_loss")
      tf.debugging.check_numerics(total_loss, message="checking total_loss")
      for i,g in enumerate(grads):
        tf.debugging.check_numerics(g, message="checking grads, grad for" + self.vars[i].name)
      for i,v in enumerate(self.vars):
        tf.debugging.check_numerics(v, message="checking self.vars AFTER update, var " + v.name)

      tf.debugging.assert_shapes([
        (advantage, ('N',)),
        (intrinsic_val_est, ('N')),
        (extrinsic_val_est, ('N')),
        (action_probs, ('N',)),
        (r, ('N',)),
        (policy_probs, ('N',ACTIONS)),
        (policy_logits, ('N',ACTIONS)),
        (total_loss, (1,)),
        ])
  

    return (policy_loss, value_loss, entropy_loss), grads, r, advantage
    

def probs_logits_entropy(probs, logits):

  # TODO do I need to do anything extra for stability at deterministic policies?
  ent =  tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(probs, logits, axis=-1)
  return ent
    
    
if __name__ == '__main__':

  agent = Agent();

  print('Saving model...')
  tf.saved_model.save(agent, model_savepath)
  tf.saved_model.save(agent, backup_savepath)

  save = {}
  with open(picklepath, "wb") as fp:
    pickle.dump(save, fp)
  with open(picklepath_backup, "wb") as fp:
    pickle.dump(save, fp)
    

    
    




    
