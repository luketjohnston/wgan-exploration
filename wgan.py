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

# let's work with input of size 84x110x4. DQN paper uses 84x84x4 because they crop the score, I'm not sure we should do this.
# score lets us tell by the screen how far along we are in the game (for example if we have to return to a room we already 
# have been to, a higher score would let us know it's the second time we've been there).
WIDTH = 84
HEIGHT = 110
DEPTH = 1

ENVIRONMENT = 'MontezumaRevenge-v0'

def makeEnv():
  return gym.make(ENVIRONMENT)

INPUT_SHAPE = [84,110,1]

LR_CRITIC = 0.00001
LR_GEN = 0.000001

ADAM_PARAMS = {'learning_rate': LR_CRITIC, 'beta_1': 0, 'beta_2': 0.9}

GRAD_LAMBDA = 10

FEATURE_SIZE = 64
HIDDEN_NEURONS=128

DECODE_ACTIVATION = tf.nn.relu

DECODE_FILTER_SIZES = [9, 5, 5, 3]
DECODE_CHANNELS =     [32,32,32,DEPTH]

#DECODE_FILTER_SIZES = []
#DECODE_CHANNELS =     [DEPTH + 1]

CRIT_FILTER_SIZES = [9,5,5]
CRIT_CHANNELS =     [32,64,64]
CRIT_STRIDES =     [4,2,1]


INT_TYPE = tf.int32
FLOAT_TYPE = tf.float32
IMSPEC = tf.TensorSpec([None,WIDTH,HEIGHT,DEPTH], dtype=FLOAT_TYPE)
INTSPEC = tf.TensorSpec([], dtype=INT_TYPE)

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'gan_save')
picklepath = os.path.join(model_savepath, 'gan.pickle')

def restrict_unit_interval(x):
  #return 1 - tf.exp(tf.pow(x,2))
  
  # the multiplier ensures the network won't just learn very large 
  # or very small values for x. Keeps x in area where gradient is meaninful
  return tf.nn.sigmoid(x)
  #return tf.nn.sigmoid(x) * 1.2 - 0.1

def getConvOutputSize(w,h,filtersize, channels, stride):
  # padding if necessary
  w = math.ceil(w / stride)
  h = math.ceil(h / stride)
  return w,h,channels

class WGAN(Model):
  def __init__(self):
    super(WGAN, self).__init__()
    self.vars = []
    self.critic_vars = []
    self.gen_vars = []


    mvars = []
    size = (WIDTH,HEIGHT,DEPTH)
    for (f, c, s) in zip(CRIT_FILTER_SIZES, CRIT_CHANNELS, CRIT_STRIDES):
      # first conv layer
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='disc_conv'))
      size = getConvOutputSize(size[0], size[1], f, c, s)


    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(size[0]*size[1]*size[2],HIDDEN_NEURONS)), name='cw3'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='cb3'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,1)), name='cw4'))

    self.critic_vars = mvars
    self.vars += mvars
    

    mvars = []
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(FEATURE_SIZE,HIDDEN_NEURONS)), name='w3'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='b3'))

    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,WIDTH*HEIGHT*DECODE_CHANNELS[0])), name='w4'))
    mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(WIDTH*HEIGHT*DECODE_CHANNELS[0],)), name='b4'))

    size = (WIDTH,HEIGHT,DECODE_CHANNELS[0])
    for (f, c) in zip(DECODE_FILTER_SIZES, DECODE_CHANNELS[1:]):
      # first conv layer
      stride = 1
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='decode_conv'))
      size = getConvOutputSize(size[0], size[1], f, c, stride)

    self.gen_vars = mvars
    self.vars += mvars

    opt = tf.keras.optimizers.Adam
    self.critic_opt = opt(**ADAM_PARAMS)
    self.gen_opt = opt(**ADAM_PARAMS)



  @tf.function(input_signature=(tf.TensorSpec(shape=None, dtype=tf.int32),))
  def genLoss(self, batch_size):
    images = self.generate(batch_size)
    scores = self.critic(images)
    loss = -1.0 * tf.reduce_mean(scores, 1) # why dim 1 here??
    

    return tf.squeeze(loss)
  

  @tf.function(input_signature=(tf.TensorSpec(shape=None, dtype=tf.int32),))
  def generate(self, num_images):
    encodings = tf.random.normal([num_images, FEATURE_SIZE])
    decodings = self.decode(encodings)
    return self.image(decodings)

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def critic(self, image):
    # fully connected module
    mvars = self.critic_vars
    x = image
    for i in range(len(CRIT_CHANNELS)):
      filt = mvars[i]
      stride = CRIT_STRIDES[i]
      x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
      x = tf.nn.leaky_relu(x)


    vi = len(CRIT_CHANNELS)
    w, b = mvars[vi:vi+2]
    x = tf.keras.layers.Flatten()(x)
    x = tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w = mvars[vi+2]
    x = tf.einsum('ba,ao->bo',x,w)
    return x

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, FEATURE_SIZE]),))
  def decode(self, encoding):
    mvars = self.gen_vars

    W,b = mvars[:2]
    x = DECODE_ACTIVATION(tf.einsum('be,eh->bh',encoding,W) + b)
    W,b = mvars[2:4]
    x = (tf.einsum('bh,hi->bi',x,W) + b)
    x = tf.reshape(x, (-1,WIDTH,HEIGHT,DECODE_CHANNELS[0]))
    vi = 4
    for i in range(len(DECODE_CHANNELS)-1):
      x = DECODE_ACTIVATION(x)
      filt = mvars[i + vi]
      stride = 1
      x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
      
    x = tf.reshape(x, [-1, WIDTH, HEIGHT, DECODE_CHANNELS[-1]]) 
    x = restrict_unit_interval(x)
    #x = 2.0 * tf.nn.sigmoid(x) - 1.0

    return x

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  def criticLoss(self, real_images, gen_images):


    real_score = tf.reduce_mean(self.critic(real_images))
    fake_score = tf.reduce_mean(self.critic(gen_images))

    eps = tf.random.uniform([], dtype=FLOAT_TYPE)
    interpolated_images = eps * real_images + (1 - eps) * gen_images
    interpolated_score = self.critic(interpolated_images)

    grad_penalty = tf.gradients(interpolated_score, interpolated_images)[0]
    grad_penalty = tf.sqrt(tf.reduce_sum(tf.square(grad_penalty), axis=[1,2,3]))
    grad_penalty = tf.reduce_mean((grad_penalty - 1)**2)
    grad_penalty *= GRAD_LAMBDA

    loss = -1.0 * (real_score - fake_score) + grad_penalty

    return tf.squeeze(loss), real_score, fake_score

  @tf.function(input_signature=(IMSPEC,INTSPEC))
  def trainBoth(self, real_images, num_fake):
    gen_images = self.generate(num_fake)
    critic_loss, real_score, fake_score = self.criticLoss(real_images, gen_images)
    gen_loss = -1.0 * fake_score

    critic_grads = tf.gradients(critic_loss, self.critic_vars)
    self.critic_opt.apply_gradients(zip(critic_grads, self.critic_vars))

    gen_grads = tf.gradients(gen_loss, self.gen_vars)
    self.gen_opt.apply_gradients(zip(gen_grads, self.gen_vars))

    return critic_loss, gen_loss

  @tf.function(input_signature=(IMSPEC,INTSPEC))
  def trainCritic(self, real_images, num_fake):
    gen_images = self.generate(num_fake)
    critic_loss, real_score, fake_score = self.criticLoss(real_images, gen_images)
    gen_loss = -1.0 * fake_score
    critic_grads = tf.gradients(critic_loss, self.critic_vars)
    self.critic_opt.apply_gradients(zip(critic_grads, self.critic_vars))
    return critic_loss


  @tf.function(input_signature=(INTSPEC,))
  def trainGen(self, num_fake):
    gen_images = self.generate(num_fake)
    scores = self.critic(gen_images)
    gen_loss = -1.0 * tf.reduce_mean(scores)
    gen_grads = tf.gradients(gen_loss, self.gen_vars)
    self.gen_opt.apply_gradients(zip(gen_grads, self.gen_vars))
    return gen_loss



  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DECODE_CHANNELS[-1]]),))
  def image(self, decoding):
    return decoding

    

  #@tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  #def __call__(self, data):
  #  encoding = self.encode(data)
  #  decoding = self.decode(encoding)
  #  image = self.image(decoding)

  #  return image


if __name__ == '__main__':
  wgan = WGAN();

  print('Saving model...')
  tf.saved_model.save(wgan, model_savepath)

  save = {}
  with open(picklepath, "wb") as fp:
    pickle.dump(save, fp)
    

