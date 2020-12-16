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

LEARNING_RATE = 0.000001
ENCODING_SIZE = 256
FILTERS = [8,8,8,8]
CHANNELS = [32,64,64,64]
STRIDES = [2,2,1,1]

BATCH_SIZE = 128

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
    self.conv_vars = []
    self.dense_vars = []
    self.decode_conv_vars = []
    self.decode_dense_vars = []
    self.strides = []
    size = (WIDTH, HEIGHT, DEPTH)
    self.conv_sizes = [size]
    
    for (f, c, s) in zip(FILTERS, CHANNELS, STRIDES):
      # first conv layer
      self.conv_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c))))
      self.strides.append(s)
      size = getConvOutputSize(size[0], size[1], f, c, s)
      self.conv_sizes.append(size)

    self.conv_out_flat_size = size[0]*size[1]*size[2]
    print('size')
    print(size)

    # first fully connected layer
    # TODO experiment with size of this layer
    self.dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(self.conv_out_flat_size,256))))
    self.dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,))))

    # output layer
    self.dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,ENCODING_SIZE))))
    self.dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,))))


    # fully connected 1
    self.decode_dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE//2,256))))
    self.decode_dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,))))
    # fully connected 2, output is reshaped to image and then resized upward
    self.decode_dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(256,self.conv_out_flat_size))))
    self.decode_dense_vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(self.conv_out_flat_size,))))

    c0 = 4
    decode_conv_vars = []
    for (f, c, s) in zip(FILTERS, CHANNELS, STRIDES):
      # first conv, stride is 1, output is resized up to original image size
      decode_conv_vars.insert(0, tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,c,c0))))
      print((f,f,c,c0))
      c0 = c
    self.decode_conv_vars += decode_conv_vars

    self.all_vars = self.conv_vars + self.dense_vars + self.decode_conv_vars + self.decode_dense_vars



  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def encode(self, x):
    for i in range(len(FILTERS)):
      x = tf.nn.conv2d(x, self.conv_vars[i], self.strides[i], 'SAME', name=None)
      x = tf.nn.relu(x)

    # collapse last dimensions and apply first linear transformation
    x = tf.reshape(x, [-1,self.conv_out_flat_size])
    x = tf.einsum('bi,io->bo', x, self.dense_vars[0]) + self.dense_vars[1]
    x = tf.nn.relu(x)
    # second linear transformation
    x = tf.einsum('bi,io->bo', x, self.dense_vars[2]) + self.dense_vars[3]
    means_and_sigmas = tf.reshape(x, (-1,ENCODING_SIZE//2,2))

    return means_and_sigmas

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,ENCODING_SIZE//2, 2]),))
  def sample(self, means_and_sigmas):
    means = means_and_sigmas[:,:,0]
    sigmas = means_and_sigmas[:,:,1]
    sample = means + tf.random.normal(shape=[means.shape[1], ENCODING_SIZE//2]) * sigmas
    return sample

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, ENCODING_SIZE//2]),))
  def decode(self, sample):
    # first linear layer
    x = tf.einsum('bi,io->bo', sample, self.decode_dense_vars[0]) + self.decode_dense_vars[1]
    x = tf.nn.relu(x)
    x = tf.einsum('bi,io->bo', x, self.decode_dense_vars[2]) + self.decode_dense_vars[3]
    # reshape to look like image with channels
    x = tf.reshape(x, [-1] + list(self.conv_sizes[-1])) 
    # resize upward to the size of the second to last convolution(which was the first)

    for s,v in zip(self.conv_sizes[:-1][::-1], self.decode_conv_vars):
      x = tf.nn.relu(x)
      x = tf.image.resize(x, s[:-1])
      x = tf.nn.conv2d(x, v, 1, 'SAME') # always stride 1 on decode? TODO

    # TODO Should I instead restrict image inputs and outputs to (0,1) to compare, so loss is smaller?
    x = 255 * tf.nn.sigmoid(x) # restrict logits to (0,1), and then multiply by 255, since it's an image.
    return x

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def loss(self, data):
    # dimensions (1,2,3) include all but batch dimension
    means_and_sigmas = self.encode(data)
    means = means_and_sigmas[:,:,0]
    sigmas = means_and_sigmas[:,:,1]
    mu2 = tf.math.pow(means, 2)
    sigma2 = tf.math.pow(sigmas, 2)
    self.kl_divergence = -1.0 *  tf.reduce_sum((1.0 + tf.math.log(sigma2) - mu2 - sigma2))
    # TODO: should we use tensorflow's kl divergence function?
    # source: https://arxiv.org/pdf/1312.6114.pdf (VAE paper)
    sample = self.sample(means_and_sigmas)
    decoding = self.decode(sample)

    MSE_WEIGHT = 1.0
    # TODO add this back in. For now, just want to get autoencoder working
    KL_WEIGHT = 0.0
    myloss =  MSE_WEIGHT * tf.reduce_mean(tf.keras.losses.MSE(data, decoding), (0,1,2)) + KL_WEIGHT * self.kl_divergence

    return myloss

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def autoencode(self, state):
    means_and_sigmas = self.encode(state)
    sample = self.sample(means_and_sigmas)
    return self.decode(sample)

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,WIDTH,HEIGHT,DEPTH]),))
  def __call__(self, data):
    return self.autoencode(data)


if __name__ == '__main__':
  encoder = Encoder();

  # have to do this once before saving model maybe?
  env = gym.make('MontezumaRevenge-v0')
  opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

  state1 = env.reset()
  state2 = env.step(env.action_space.sample())[0]
  state3 = env.step(env.action_space.sample())[0]
  state4 = env.step(env.action_space.sample())[0]
  statelist = [state1, state2, state3, state4]
  
  statelist = [tf.image.rgb_to_grayscale(s) for s in statelist]
  statelist = [tf.image.resize(s,(84,110)) for s in statelist] #TODO does method of downsampling matter?
  
  state = tf.stack(statelist, -1)
  state = tf.squeeze(state)

  batchlist = []
  for i in range(1):
    statelist.pop(0)
    observation = env.step(env.action_space.sample())[0]

    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation,(84,110)) #TODO does method of downsampling matter?

    statelist.append(observation)

    state = tf.stack(statelist, -1)
    state = tf.squeeze(state)
    batchlist.append(state)

  batch = tf.stack(batchlist, 0)
  with tf.GradientTape() as tape:
    loss = encoder.loss(batch)
    all_vars = tape.watched_variables()
    gradients = tape.gradient(loss, all_vars)
    opt.apply_gradients(zip(gradients, all_vars))
  
  print('Saving model...')
  tf.saved_model.save(encoder, model_savepath)


  losses = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
    

