import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import math
import os
from tensorflow.keras import Model
import gym
import pickle

#import tensorflow_probability as tfp
#tfd = tfp.distributions


# TODO it seems like the decoding is predicting a state a bit into the future. Investigate this?
# perhaps it is predicting a constant state rather than a unique one for each of the 4 timesteps?


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



# For some reason this is necessary to prevent error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def restrict_unit_interval(x):
  #return 1 - tf.exp(tf.pow(x,2))
  
  # the multiplier ensures the network won't just learn very large 
  # or very small values for x. Keeps x in area where gradient is meaninful
  #return tf.nn.sigmoid(x)
  return tf.nn.sigmoid(x) * 1.1





# let's work with input of size 84x110x4. DQN paper uses 84x84x4 because they crop the score, I'm not sure we should do this.
# score lets us tell by the screen how far along we are in the game (for example if we have to return to a room we already 
# have been to, a higher score would let us know it's the second time we've been there).
WIDTH = 84
HEIGHT = 110
DEPTH = 1

MODULES = 2

ENCODING_SIZE = 4


dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'encoder.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')

ENCODE_FILTER_SIZES = [3, 3, 3]
ENCODE_CHANNELS =     [16,16,16]

DECODE_FILTER_SIZES = [3, 3, 16]
DECODE_CHANNELS =     [16,16,DEPTH+1]


HIDDEN_NEURONS = 32

def getConvOutputSize(w,h,filtersize, channels, stride):
  # padding if necessary
  w = math.ceil(w / stride)
  h = math.ceil(h / stride)
  return w,h,channels



class Encoder(Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.module_vars = []
    self.vars = []




    for m in range(MODULES):
      mvars = []

      size = (WIDTH,HEIGHT,DEPTH)
      for (f, c) in zip(ENCODE_FILTER_SIZES, ENCODE_CHANNELS):
        # first conv layer
        stride = 1
        mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='encode_conv'))
        size = getConvOutputSize(size[0], size[1], f, c, stride)


      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HEIGHT*WIDTH*ENCODE_CHANNELS[-1],HIDDEN_NEURONS)), name='w1'))
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='b1'))
      # need these "1" size axis in the b vars so that we can broadcast to the batch dimension

      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ENCODING_SIZE)), name='w2'))
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,)), name='b2'))

      # output layer

      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,HIDDEN_NEURONS)), name='w3'))
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='b3'))

      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,WIDTH*HEIGHT)), name='w4'))
      mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(WIDTH*HEIGHT,)), name='b4'))

      size = (WIDTH,HEIGHT,1)
      for (f, c) in zip(DECODE_FILTER_SIZES, DECODE_CHANNELS):
        # first conv layer
        stride = 1
        mvars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='decode_conv'))
        size = getConvOutputSize(size[0], size[1], f, c, stride)
      self.module_vars += [mvars]
      self.vars += mvars


    self.background = tf.Variable(tf.initializers.GlorotNormal()(shape=(1,WIDTH,HEIGHT,DEPTH,)), name='b0')
    
    self.vars += [self.background]



  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def encode(self, data):

    encodings = []
    for m in range(MODULES):
      mvars = self.module_vars[m]
      x = data
      for i in range(len(ENCODE_CHANNELS)):
        filt = mvars[i]
        stride = 1
        x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
        x = tf.nn.leaky_relu(x)

      # collapse last dimensions and apply first linear transformation
      flattened_size = HEIGHT * WIDTH * ENCODE_CHANNELS[-1]
      x = tf.reshape(x, [-1,flattened_size])

      vi = len(ENCODE_FILTER_SIZES)
      x = tf.einsum('bi,io->bo', x, mvars[vi]) + mvars[vi+1]
      x = tf.nn.leaky_relu(x)
      x = tf.einsum('bi,io->bo', x, mvars[vi+2]) + mvars[vi+3]
      x = tf.nn.leaky_relu(x)
      encodings.append(x)
    return tf.stack(encodings)

  @tf.function(input_signature=(tf.TensorSpec(shape=[MODULES,None, ENCODING_SIZE]),))
  def decode(self, encoding):

    decodings = []
    for m in range(MODULES):
      mvars = self.module_vars[m]
      x = encoding[m,:,:]
      # first linear layer
      vi = len(ENCODE_FILTER_SIZES) + 4
      x = tf.einsum('bi,io->bo', x, mvars[vi]) + mvars[vi+1]
      x = tf.nn.leaky_relu(x)
      x = tf.einsum('bi,io->bo', x, mvars[vi+2]) + mvars[vi+3]

      # TODO leaky_relu here or no?

      # TODO should we change the initial channels? probably...
      x = tf.reshape(x, (-1,WIDTH,HEIGHT,1))

      for i in range(len(DECODE_CHANNELS)):
        filt = mvars[i + vi + 4]
        stride = 1
        x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
        x = tf.nn.leaky_relu(x)
        

      x = tf.reshape(x, [-1, WIDTH, HEIGHT, DECODE_CHANNELS[-1]]) 

      #x = tf.nn.sigmoid(x)
      x = restrict_unit_interval(x)
      decodings.append(x)

    return tf.stack(decodings)

  @tf.function(input_signature=(tf.TensorSpec(shape=[MODULES,None,WIDTH,HEIGHT,DECODE_CHANNELS[-1]]),))
  def image(self, decoding):

    def scanF(acc, elem):
      x = tf.stop_gradient(acc) * (1 - elem[:,:,:,-1:]) + elem[:,:,:,:-1] * elem[:,:,:,-1:]
      return tf.clip_by_value(x, 0., 1.)

    init = tf.zeros((tf.shape(decoding)[1],WIDTH,HEIGHT,DEPTH))
    init += self.background

    image = tf.scan(scanF, decoding, initializer=init)
    return image[-1,:,:,:,:]
      

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def loss_from_beginning(self, data):
    losses = []
    reconstruction = self.vars[-1]
    losses.append(tf.reduce_mean(tf.keras.losses.MSE(reconstruction, data)))

    for m in range(MODULES):
      mvars = self.module_vars[m]
      x = data
      for i in range(len(ENCODE_CHANNELS)):
        filt = mvars[i]
        stride = 1
        x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
        x = tf.nn.leaky_relu(x)

      # collapse last dimensions and apply first linear transformation
      flattened_size = HEIGHT * WIDTH * ENCODE_CHANNELS[-1]
      x = tf.reshape(x, [-1,flattened_size])

      vi = len(ENCODE_FILTER_SIZES)
      x = tf.einsum('bi,io->bo', x, mvars[vi]) + mvars[vi+1]
      x = tf.nn.leaky_relu(x)
      x = tf.einsum('bi,io->bo', x, mvars[vi+2]) + mvars[vi+3]
      x = tf.nn.leaky_relu(x)

      # DECODING
      # first linear layer
      vi = len(ENCODE_FILTER_SIZES) + 4
      x = tf.einsum('bi,io->bo', x, mvars[vi]) + mvars[vi+1]
      x = tf.nn.leaky_relu(x)
      x = tf.einsum('bi,io->bo', x, mvars[vi+2]) + mvars[vi+3]

      x = tf.reshape(x, (-1,WIDTH,HEIGHT,1))

      for i in range(len(DECODE_CHANNELS)):
        filt = mvars[i + vi + 4]
        stride = 1
        x = tf.nn.conv2d(x,filt,stride,'SAME',name=None)
        x = tf.nn.leaky_relu(x)
        
      x = tf.reshape(x, [-1, WIDTH, HEIGHT, DECODE_CHANNELS[-1]]) 

      # REALISTICALLY I should output a mask and a change,
      # and mix accordingly, so never go outside range (0,1)
      #x = 2 * tf.nn.sigmoid(x) - 1 # restrict logits to (-1,1),
      #x = tf.nn.sigmoid(x) # restrict logits to (-1,1),
      x = restrict_unit_interval(x)
      reconstruction = tf.stop_gradient(reconstruction * (1 - x[:,:,:,-1:])) + x[:,:,:,:-1] * x[:,:,:,-1:]
      losses.append(tf.reduce_mean(tf.keras.losses.MSE(reconstruction,data)))
      reconstruction = tf.clip_by_value(reconstruction, 0., 1.)

    return losses

  #@tf.function(input_signature=(tf.TensorSpec(shape=[MODULES,None,WIDTH,HEIGHT,DEPTH]), 
  #                              tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  #def loss_from_decoding(self, decoding, data):
  #  # TODO investigate leaky vs normal relu

  #  def scanF(acc, elem):
  #    return tf.stop_gradient(acc) + elem

  #  init = tf.zeros(tf.shape(data))
  #  approximations = tf.scan(scanF, decoding, initializer=init)
  #  losses = tf.keras.losses.MSE(approximations, data)

  #  return losses


  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def __call__(self, data):
    encoding = self.encode(data)
    decoding = self.decode(encoding)
    image = self.image(decoding)

    return image


if __name__ == '__main__':
  encoder = Encoder();

  
  print('Saving model...')
  tf.saved_model.save(encoder, model_savepath)


  losses = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
    

