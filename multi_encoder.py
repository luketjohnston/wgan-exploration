import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import os
from tensorflow.keras import Model
import gym
import pickle

#import tensorflow_probability as tfp
#tfd = tfp.distributions


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
DEPTH = 4

ENCODING_SIZE = 512
FILTERS = [4,4,4,4]
CHANNELS = [16,16,16,16]
STRIDES = [2,2,2,1]


dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'encoder.mod')
loss_savepath = os.path.join(dir_path, 'loss.pickle')


# FOR NOW, I believe this assumes the filtersize is divisible by the stride
def getConvOutputSize(w,h,filtersize, channels, stride):
  # padding if necessary
  if ((h - filtersize) % stride): h += (stride -((h - filtersize) % stride))
  if ((w - filtersize) % stride): w += (stride - ((w - filtersize) % stride))
  return ((w - filtersize) // stride + filtersize // stride, (h - filtersize) // stride  + filtersize // stride, channels)

  
   

class Encoder(Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.adjustment_conv = []
    self.adjustment_dense = []
    self.decode_dense_vars = []
    self.strides = []
    size = (WIDTH, HEIGHT, DEPTH)
    self.conv_sizes = [size]

    self.background_dense = []
    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(WIDTH * HEIGHT * DEPTH,10)), name='back1'))
    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(10,)), name='back2'))
    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(10,10)), name='back3'))
    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(10,)), name='back4'))

    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(5,10)), name='back_decode1'))
    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(10,)), name='back_decode2'))
    self.background_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(10,WIDTH*HEIGHT)), name='back_decode3'))

    for (f, c, s) in zip(FILTERS, CHANNELS, STRIDES):
      # first conv layer
      print('creating with size')
      print((f,f,size[2],c))
      self.adjustment_conv.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='conv'))
      self.strides.append(s)
      size = getConvOutputSize(size[0], size[1], f, c, s)
      self.conv_sizes.append(size)

    self.conv_out_flat_size = size[0]*size[1]*size[2]

    # first fully connected layer
    # TODO experiment with size of this layer
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(self.conv_out_flat_size,256)), name='w0'))
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,)), name='b0'))

    # output layer
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,ENCODING_SIZE)), name='w1'))
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,)), name='b1'))


    # fully connected 1
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE//2,256)), name='decode_w2'))
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,)), name='decode_b2'))
    # fully connected 2, output is reshaped to image and then resized upward
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,self.conv_out_flat_size)), name='decode_w3'))
    self.adjustment_dense.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(self.conv_out_flat_size,)), name='decode_b3'))

    c0 = 4
    decode_conv_vars = []
    for (f, c, s) in zip(FILTERS, CHANNELS, STRIDES):
      # first conv, stride is 1, output is resized up to original image size
      decode_conv_vars.insert(0, tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,c0,c))))
      c0 = c
    self.adjustment_conv += decode_conv_vars

    self.background_vars = self.background_dense
    self.adjustment_vars = self.adjustment_conv + self.adjustment_dense
    self.all_vars = self.adjustment_conv + self.adjustment_dense + self.background_dense

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def encode_background(self, x):
    x = tf.reshape(x, [-1,WIDTH * HEIGHT * DEPTH])
    x = tf.einsum('bi,io->bo', x, self.background_dense[0]) + self.background_dense[1]
    x = tf.nn.relu(x)
    x = tf.einsum('bi,io->bo', x, self.background_dense[2]) + self.background_dense[3]
    means_and_sigmas = tf.reshape(x, (-1, 10//2, 2))
    return means_and_sigmas



  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def encode_adjustment(self, x):
    for i in range(len(FILTERS)):
      x = tf.nn.conv2d(x, self.adjustment_conv[i], self.strides[i], 'SAME', name=None)
      x = tf.nn.relu(x)

    # collapse last dimensions and apply first linear transformation
    x = tf.reshape(x, [-1,self.conv_out_flat_size])
    x = tf.einsum('bi,io->bo', x, self.adjustment_dense[0]) + self.adjustment_dense[1]
    x = tf.nn.relu(x)
    # second linear transformation
    x = tf.einsum('bi,io->bo', x, self.adjustment_dense[2]) + self.adjustment_dense[3]
    means_and_sigmas = tf.reshape(x, (-1,ENCODING_SIZE//2,2))

    return means_and_sigmas

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,None,2]),))
  def sample(self, means_and_sigmas):
    means = means_and_sigmas[:,:,0]
    sigmas = means_and_sigmas[:,:,1]
    sample = means + tf.random.normal(shape=tf.shape(means)) * sigmas
    return sample

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, 10//2]),))
  def background(self, sample):
    # first "step" is dense layer directly from encoding to output image,
    # and is stacked to produce the 4-timestep output. This allows the model
    # to easily learn the "background" image (the room)
    background = tf.einsum('bi,io->bo', sample, self.background_dense[4]) + self.background_dense[5]
    background = tf.nn.relu(background)
    background = tf.einsum('bi,io->bo', background, self.background_dense[6])
    background = tf.reshape(background, [-1, WIDTH, HEIGHT, 1])
    background = tf.concat([background, background, background, background], -1)

    return 255 * tf.nn.sigmoid(background)
    

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, ENCODING_SIZE//2]),))
  def adjustment(self, sample):


    # first linear layer
    x = tf.einsum('bi,io->bo', sample, self.adjustment_dense[4]) + self.adjustment_dense[5]
    x = tf.nn.relu(x)
    x = tf.einsum('bi,io->bo', x, self.adjustment_dense[6]) + self.adjustment_dense[7]
    # reshape to look like image with channels
    x = tf.reshape(x, [-1] + list(self.conv_sizes[-1])) 
    # resize upward to the size of the second to last convolution(which was the first)

    for s,v,stride in zip(self.conv_sizes[:-1][::-1], self.adjustment_conv[len(FILTERS):], STRIDES[::-1]):
      x = tf.nn.relu(x)
      #x = tf.image.resize(x, s[:-1])
      print('transpose output shape')
      x = tf.nn.conv2d_transpose(x, v, s[:-1], [1,stride,stride,1], padding='SAME')
      print(x.shape)
      
      #x = tf.nn.conv2d(x, v, 1, 'SAME') # always stride 1 on decode? TODO

    x = tf.nn.sigmoid(x) # restrict logits to (0,1),

    # TODO Should I instead restrict image inputs and outputs to (0,1) to compare, so loss is smaller?
    x = 255.0*x
    return x

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, 10//2]),tf.TensorSpec(shape=[None, ENCODING_SIZE//2])))
  def decode(self, background_sample, adjustment_sample):
    background = self.background(background_sample)
    adjustment = self.adjustment(adjustment_sample)
    return background + adjustment

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),
                                tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  def background_loss(self, background, data):
    return tf.reduce_mean(tf.keras.losses.MSE(data, background))

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, ENCODING_SIZE//2]), 
                                tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),
                                tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH])))
  def adjustment_loss(self, sample, background, data):


    adjustment = self.adjustment(sample)

    # regularization term to make sure adjustment doesn't get stuck in local minima at 0
    # not using currently TODO
    reg = tf.reduce_mean(tf.math.pow(adjustment, 2))
    reg = tf.math.abs(100 - reg)

    diff = tf.nn.relu(data - tf.stop_gradient(background))
    adj_loss = tf.reduce_mean(tf.keras.losses.MSE(diff, adjustment))
    #adj_loss += reg
    return adj_loss

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,None, 2]),))
  def kl_loss(self, means_and_sigmas):
    means = means_and_sigmas[:,:,0]
    sigmas = means_and_sigmas[:,:,1]
    mu2 = tf.math.pow(means, 2)
    sigma2 = tf.math.pow(sigmas, 2)
    kl_divergence = -1.0 *  tf.reduce_sum((1.0 + tf.math.log(sigma2) - mu2 - sigma2))
    # TODO: should we use tensorflow's kl divergence function?
    # source: https://arxiv.org/pdf/1312.6114.pdf (VAE paper)

    return kl_divergence

  #@tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  #def autoencode(self, state):
  #  means_and_sigmas = self.encode(state)
  #  sample = self.sample(means_and_sigmas)
  #  return self.decode(sample)

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def __call__(self, data):
    background_means_and_sigmas = self.encode_background(data)
    adjustment_means_and_sigmas = self.encode_adjustment(data)
    background_sample = self.sample(background_means_and_sigmas)
    adjustment_sample = self.sample(adjustment_means_and_sigmas)

    return self.decode(background_sample, adjustment_sample)


if __name__ == '__main__':
  encoder = Encoder();

  
  print('Saving model...')
  tf.saved_model.save(encoder, model_savepath)


  losses = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
    

