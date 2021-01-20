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

# let's work with input of size 84x110x4. DQN paper uses 84x84x4 because they crop the score, I'm not sure we should do this.
# score lets us tell by the screen how far along we are in the game (for example if we have to return to a room we already 
# have been to, a higher score would let us know it's the second time we've been there).
WIDTH = 84
HEIGHT = 110
DEPTH = 1

MODULES = 1

FEATURE_SIZE = 32
HIDDEN_NEURONS=64

DECODE_ACTIVATION = tf.nn.relu

USE_BACKGROUND=True


DECODE_FILTER_SIZES = [7, 5, 5, 5]
DECODE_CHANNELS =     [32,64,128,DEPTH]

#DECODE_FILTER_SIZES = []
#DECODE_CHANNELS =     [DEPTH + 1]

CRIT_FILTER_SIZES = [5, 7, 7, 7, 7]
CRIT_CHANNELS =     [32,64,64,128,128]
CRIT_STRIDES =     [2,2,2,2,1]

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'gan.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')

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

class GAN(Model):
  def __init__(self):
    super(GAN, self).__init__()
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


  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  def criticLoss(self, real_images, gen_images):


    real_score = tf.reduce_sum(self.critic(real_images))
    fake_score = tf.reduce_sum(self.critic(gen_images))

    # to make computing gradients easier
    total_score = real_score - fake_score # shape (None,)
  
    # apparently tf.gradients just computes as if total_score was summed. 
    # if we want gradients with respect to multiple things, need to give
    # a list as input.
    real_grad_penalty = tf.gradients(total_score, real_images)[0]
    real_grad_penalty = tf.reshape(real_grad_penalty, (-1, WIDTH * HEIGHT * DEPTH))
    real_grad_penalty = tf.norm(real_grad_penalty, axis=-1, ord=2)
    real_grad_penalty = tf.math.pow(real_grad_penalty-1.0, 2)
    real_grad_penalty = tf.reduce_sum(real_grad_penalty)

    fake_grad_penalty = tf.gradients(total_score, gen_images)[0]
    fake_grad_penalty = tf.reshape(fake_grad_penalty, (-1, WIDTH * HEIGHT * DEPTH))
    fake_grad_penalty = tf.norm(fake_grad_penalty, axis=-1, ord=2)
    fake_grad_penalty = tf.math.pow(fake_grad_penalty-1.0, 2)
    fake_grad_penalty = tf.reduce_sum(fake_grad_penalty)

    grad_penalty = 10 * (real_grad_penalty + fake_grad_penalty)

    loss = -1.0 * (real_score - fake_score) + grad_penalty

    return tf.squeeze(loss)

  @tf.function(input_signature=(tf.TensorSpec(shape=None, dtype=tf.int32),))
  def genLoss(self, batch_size):
    images = self.generate(batch_size)
    # images has shape [MODULES, batch] + imshape
    scores = self.critic(images)
    loss = -1.0 * tf.reduce_sum(scores, 1)
    

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
  gan = GAN();

  #print('running loss')
  #encoder.loss_from_beginning(tf.zeros((16,84,110,1)))

  
  print('Saving model...')
  tf.saved_model.save(gan, model_savepath)


  criticLosses = []
  genLosses = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump((criticLosses, genLosses), fp)
    

