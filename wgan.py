import tensorflow as tf
import math
from tensorflow.keras import Model
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow_datasets as tfds
import numpy as np


DEPTH = 1

USE_BATCH_NORM = False

LR_CRITIC = 0.001
LR_GEN = LR_CRITIC

ADAM_PARAMS = {'learning_rate': LR_CRITIC, 'beta_1': 0, 'beta_2': 0.9}

ACTIVATION = tf.nn.leaky_relu
DECODE_ACTIVATION = tf.nn.relu

GRAD_LAMBDA = 1

FEATURE_SIZE = 128

DECODE_IN_SHAPE = [11,11,128]

DECODE_LAYERS = [1024,DECODE_IN_SHAPE[0]*DECODE_IN_SHAPE[1]*DECODE_IN_SHAPE[2]]

DECODE_FILTER_SIZES = [4,4,4]
DECODE_CHANNELS =     [128,64,1]
DECODE_STRIDES =     [2,2,2]

DECONV_SHAPES = [[21,21],[42,42],[84,84]]
DECONV_PADDINGS = [[1,1],None,None]

CRIT_FILTER_SIZES = [8,4,4]
CRIT_CHANNELS =     [64,128,128]
CRIT_STRIDES =     [4,2,1]

CRIT_LAYERS = [1024,1]


INT_TYPE = tf.int32
FLOAT_TYPE = tf.float32

WIDTH,HEIGHT,DEPTH = [84,84,1]
IMSPEC = tf.TensorSpec([None,WIDTH,HEIGHT,DEPTH], dtype=FLOAT_TYPE)
BOOLSPEC = tf.TensorSpec([], dtype=tf.bool)
INTSPEC = tf.TensorSpec([], dtype=INT_TYPE)


def getConvOutputSize(w,h,filtersize, channels, stride):
  # padding if necessary
  w = math.ceil(w / stride)
  h = math.ceil(h / stride)
  return w,h,channels

class Critic(tf.keras.Model):
  def __init__(self):
    super(Critic, self).__init__()
    # make critic
    self.convs = []
    for i,(f, c, s) in enumerate(zip(CRIT_FILTER_SIZES, CRIT_CHANNELS, CRIT_STRIDES)):
      if i == 0: # first conv layer, supply input_shape kwarg
        self.convs.append(tf.keras.layers.Conv2D(c,f,s,padding='SAME',activation=ACTIVATION, input_shape=(WIDTH,HEIGHT,DEPTH)))
      else:
        self.convs.append(tf.keras.layers.Conv2D(c,f,s,padding='SAME',activation=ACTIVATION))

    self.dense = []
    for i,l in enumerate(CRIT_LAYERS):
      act = ACTIVATION
      if i == len(CRIT_LAYERS) - 1:
        act = None
      self.dense.append(tf.keras.layers.Dense(l,activation=act))
    opt = tf.keras.optimizers.Adam
    self.opt = opt(**ADAM_PARAMS)

  
  @tf.function(input_signature=(IMSPEC,))
  def call(self, image):
    x = image
    for conv in self.convs:
      x = conv(x)
    x = tf.keras.layers.Flatten()(x)
    for layer in self.dense:
      x = layer(x)
    x = tf.squeeze(x,-1)
    return x



class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()

    self.dense = []
    self.denseBN = []
    for l in DECODE_LAYERS:
      # don't need bias with batch normalization
      self.dense.append(tf.keras.layers.Dense(l,activation=DECODE_ACTIVATION,use_bias=False))
      self.denseBN.append(tf.keras.layers.BatchNormalization())


    self.convT = []
    self.convT_BN = []
    for i,(f, c, s) in enumerate(zip(DECODE_FILTER_SIZES, DECODE_CHANNELS, DECODE_STRIDES)):
      if i == len(DECODE_FILTER_SIZES) - 1:
        act = tf.nn.sigmoid
        use_bias = True
      else:
        act = DECODE_ACTIVATION
        self.convT_BN.append(tf.keras.layers.BatchNormalization(axis=3))
        use_bias = False
      self.convT.append(tf.keras.layers.Conv2DTranspose(c,f,s,padding='SAME',activation=act,use_bias=use_bias,output_padding=DECONV_PADDINGS[i]))

    opt = tf.keras.optimizers.Adam
    self.opt = opt(**ADAM_PARAMS)


  @tf.function(input_signature=(INTSPEC,BOOLSPEC))
  def call(self, num_images, is_training=True):
    #TODO add hyperparam for whether to use normal or uniform sampling
    # uniform seems to work slightly better than normal on mnist
    x = tf.random.uniform([num_images, FEATURE_SIZE],-1,1) 
    for layer, bn in zip(self.dense, self.denseBN):
      x = layer(x)
      if USE_BATCH_NORM: 
        x = bn(x, training=is_training)
    x = tf.reshape(x, [-1] + DECODE_IN_SHAPE)
    for i,conv_transpose in enumerate(self.convT):
      x = conv_transpose(x)
      if i < len(self.convT) - 1 and USE_BATCH_NORM:
        x = self.convT_BN[i](x, training=is_training)
    return x


class WGAN(Model):
  def __init__(self):
    super(WGAN, self).__init__()

    self.critic = Critic()
    self.gen = Generator()


  @tf.function(input_signature=(IMSPEC,INTSPEC))
  def trainCritic(self, real_images, num_fake):
    gen_images = self.gen(num_fake,True)
    critic_loss, real_score, fake_score = self.criticLoss(real_images, gen_images)
    critic_grads = tf.gradients(critic_loss, self.critic.trainable_weights)
    self.critic.opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
    return critic_loss


  @tf.function(input_signature=(INTSPEC,))
  def trainGen(self, num_fake):
    gen_images = self.gen(num_fake,True)
    scores = self.critic.call(gen_images)
    gen_loss = -1.0 * tf.reduce_mean(scores)
    gen_grads = tf.gradients(gen_loss, self.gen.trainable_weights)
    self.gen.opt.apply_gradients(zip(gen_grads, self.gen.trainable_weights))
    return gen_loss


  @tf.function(input_signature=(IMSPEC,IMSPEC))
  def criticLoss(self, real_images, gen_images):


    real_score = tf.reduce_mean(self.critic(real_images))
    fake_score = tf.reduce_mean(self.critic(gen_images))

    batch_size = tf.shape(real_images)[0:1]
    
    eps = tf.random.uniform(batch_size, dtype=FLOAT_TYPE)
    real_weighted = tf.einsum('b,bhwc->bhwc',eps, real_images)  # einsum is clearer than broadcasting IMO
    fake_weighted = tf.einsum('b,bhwc->bhwc',(1 - eps), real_images) 
    interpolated_images = real_weighted + fake_weighted
    interpolated_score = self.critic.call(interpolated_images)

    grad_penalty = tf.gradients(interpolated_score, [interpolated_images])[0]
    grad_penalty = tf.sqrt(tf.reduce_sum(tf.square(grad_penalty), axis=[1,2,3]))
    grad_penalty = tf.reduce_mean(tf.pow((grad_penalty - 1.0),2))
    grad_penalty *= GRAD_LAMBDA

    loss = -1.0 * (real_score - fake_score) + grad_penalty

    return tf.squeeze(loss), real_score, fake_score

  @tf.function(input_signature=(INTSPEC,))
  def generate(self, num_images):
    return self.gen(num_images, False)

  @tf.function(input_signature=(IMSPEC,))
  def score(self, images):
    return self.critic(images)
  
  

