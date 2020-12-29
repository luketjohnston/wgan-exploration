import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import math
import os
from tensorflow.keras import Model
import gym
import pickle

#import tensorflow_probability as tfp
#tfd = tfp.distributions


# TODO it seems like the adjustment is predicting a state a bit into the future. Investigate this?
# perhaps it is predicting a constant state rather than a unique one for each of the 4 timesteps?


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

ENCODING_SIZE = 64


dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'encoder.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')

ENCODE_FILTER_SIZES = [3, 3, 3]
ENCODE_CHANNELS =     [16,16,1]

DECODE_FILTER_SIZES = [3, 3, 16]
DECODE_CHANNELS =     [16,16,DEPTH]

def getConvOutputSize(w,h,filtersize, channels, stride):
  # padding if necessary
  w = math.ceil(w / stride)
  h = math.ceil(h / stride)
  return w,h,channels



class Encoder(Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.vars = []



    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(1,WIDTH,HEIGHT,DEPTH,)), name='b0'))


    size = (WIDTH,HEIGHT,DEPTH)
    for (f, c) in zip(ENCODE_FILTER_SIZES, ENCODE_CHANNELS):
      # first conv layer
      stride = 1
      self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='encode_conv'))
      size = getConvOutputSize(size[0], size[1], f, c, stride)


    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HEIGHT*WIDTH*ENCODE_CHANNELS[-1],ENCODING_SIZE)), name='w1'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,)), name='b1'))

    #self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,ENCODING_SIZE)), name='w2'))
    #self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,)), name='b2'))

    # output layer

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,64)), name='w3'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(64,)), name='b3'))

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(64,WIDTH*HEIGHT)), name='w4'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(WIDTH*HEIGHT,)), name='b4'))

    size = (WIDTH,HEIGHT,1)
    for (f, c) in zip(DECODE_FILTER_SIZES, DECODE_CHANNELS):
      # first conv layer
      stride = 1
      self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='decode_conv'))
      size = getConvOutputSize(size[0], size[1], f, c, stride)
    


    self.background_vars = self.vars[:1]
    self.adjustment_vars = self.vars[1:]



  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def background(self, x):
    return self.vars[0]

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def encode_adjustment(self, x):

    for (f, c, v) in zip(ENCODE_FILTER_SIZES, ENCODE_CHANNELS, self.vars[1:1+len(ENCODE_FILTER_SIZES)]):
      stride = 1
      x = tf.nn.conv2d(x, v, stride, 'SAME', name=None)
      x = tf.nn.leaky_relu(x)

    # collapse last dimensions and apply first linear transformation
    #flattened_size = self.conv_sizes[-1][0] * self.conv_sizes[-1][1] * self.conv_sizes[-1][2]
    flattened_size = HEIGHT * WIDTH * ENCODE_CHANNELS[-1]
    x = tf.reshape(x, [-1,flattened_size])

    vi = 1+len(ENCODE_FILTER_SIZES)
    x = tf.einsum('bi,io->bo', x, self.vars[vi]) + self.vars[vi+1]
    #x = tf.nn.leaky_relu(x)
    #x = tf.einsum('bi,io->bo', x, self.vars[vi+2]) + self.vars[vi+3]
    #x = tf.nn.leaky_relu(x)
    return x

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, ENCODING_SIZE]),))
  def decode_adjustment(self, encoding):

    # first linear layer
    vi = 1 + len(ENCODE_FILTER_SIZES) + 2
    x = tf.einsum('bi,io->bo', encoding, self.vars[vi]) + self.vars[vi+1]
    x = tf.nn.leaky_relu(x)
    x = tf.einsum('bi,io->bo', x, self.vars[vi+2]) + self.vars[vi+3]

    x = tf.reshape(x, (-1,WIDTH,HEIGHT,1))

    for (f, c, v) in zip(DECODE_FILTER_SIZES, DECODE_CHANNELS, self.vars[vi+4:vi+4+len(DECODE_FILTER_SIZES)]):
      stride = 1
      x = tf.nn.leaky_relu(x)
      x = tf.nn.conv2d(x, v, stride, 'SAME', name=None)
      


    x = tf.reshape(x, [-1, WIDTH, HEIGHT, DEPTH]) 
    x = 2 * tf.nn.sigmoid(x) - 1 # restrict logits to (-1,1),


    return x

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]), 
                                tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),
                                tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  def adjustment_loss(self, background, adjustment, data):
    # FOR SOME REASON, my attempts to learn the following without the "relu"
    # in front don't work. Possibly because I had sigmoid at the end of decode_adjustment?
    # a few attempts without the sigmoid also didn't work so not really sure what's going on
    # UPDATE: with leaky relu activations in the rest of the network, we can now learn
    # the absolute diff. TODO investigate?


    diff = (data - tf.stop_gradient(background))
    #diff = data - tf.stop_gradient(background)
    loss = tf.reduce_mean(tf.keras.losses.MSE(adjustment, diff))
    #loss = tf.reduce_mean(tf.keras.losses.MSE(adjustment, data))
    return loss


  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]), 
                                tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  def background_loss(self, background, data):

    # BCE works better than MAE
    # MSE works about the same as BCE
    #loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(data, decoding))
    #loss = tf.reduce_mean(tf.keras.losses.MAE(data, decoding))
    loss = tf.reduce_mean(tf.keras.losses.MSE(background, data))
    #adj_loss += reg
    return loss

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def __call__(self, data):
    background = self.background(data)
    encoding = self.encode_adjustment(data)
    adjustment = self.decode_adjustment(encoding)

    return background + adjustment
    #return adjustment


if __name__ == '__main__':
  encoder = Encoder();

  
  print('Saving model...')
  tf.saved_model.save(encoder, model_savepath)


  losses = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
    

