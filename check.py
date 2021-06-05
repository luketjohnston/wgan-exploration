import matplotlib
import argparse
from time import sleep
matplotlib.use('tkagg')

import tensorflow as tf
import os
from tensorflow.keras import Model
import gym

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pynput.keyboard import Key, Listener
from pynput import keyboard

import agent
import explore


parser = argparse.ArgumentParser(description="explore with |wgan critic loss| intrinsic reward")
parser.add_argument('load_model', type=str, help="path to model to load")
parser.add_argument('--control', default=False, action="store_true", help="whether user controls actions")
args = parser.parse_args()
savepath = args.load_model
actor_savepath = os.path.join(savepath, 'actor')
wgan_savepath = os.path.join(savepath, 'wgan')

with tf.device('/device:CPU:0'):
  actor = tf.saved_model.load(actor_savepath)
  wgan = tf.saved_model.load(wgan_savepath)
  
  # make environment
  env = explore.makeEnv()
  state = env.reset()
  
  action = 0
  
  [left, right, up, down, space] = [False,False,False,False,False]
  set_start = True
  set_goal = True
  
  def on_press(key):
    global left
    global right
    global up
    global down
    global space
    global set_start
    global set_goal
    if key == keyboard.KeyCode(char='a'):
      left=True
    if key == keyboard.KeyCode(char='d'):
      right=True
    if key == keyboard.KeyCode(char='w'):
      up=True
    if key == keyboard.KeyCode(char='s'):
      down=True
    if key == Key.space:
      space=True
    if key == keyboard.KeyCode(char='q'):
      set_start = True
    if key == keyboard.KeyCode(char='e'):
      set_goal = True
  
  def on_release(key):
    global left
    global right
    global up
    global down
    global space
    if key == keyboard.KeyCode(char='a'):
      left=False
    if key == keyboard.KeyCode(char='d'):
      right=False
    if key == keyboard.KeyCode(char='w'):
      up=False
    if key == keyboard.KeyCode(char='s'):
      down=False
    if key == Key.space:
      space=False
  
  def getAction():
    global left
    global right
    global up
    global down
    global space
    if space and left:
      return 12
    if space and right:
      return 11
    if space:
      return 1
    if up:
      return 2
    if right:
      return 3
    if left:
      return 4
    if down:
      return 5
    return 0
    
  
  ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
  
  
  # Collect events until released
  with Listener(
          on_press=on_press,
          on_release=on_release) as listener:
  
  
    start = None
    goal = None
  
    while True:
      #if set_start:
      #  start = state
      #  print('Setting start!')
      #  set_start = False
  
      #if set_goal:
      #  goal = state
      #  set_goal = False
      #  print('Distance: ') 
      #  print('start to goal: ')
      #  print(tf.math.exp(actor.distance_states(start, goal)))
      #  print('goal to start: ')
      #  print(tf.math.exp(actor.distance_states(goal, start)))
  
  
     
      state = np.expand_dims(state._force(), 0)
      policy =  actor.policy(state)
      action, _ = actor.act(policy)
  
      sleep(0.0416)
      if args.control:
        action = getAction()
      state, _, done, _ = env.step(action)
      gan_input = np.expand_dims(state._force(), 0)[...,-1:]
      print(wgan.critic(gan_input))
  
      if done: env.reset()
  
      env.render()
    
    
      #fig, axes = plt.subplots(1,2)
      #
      #original = tf.squeeze(state)
      #axes[0].imshow(original, cmap=cm.gray)
    
      #
      #plt.show()
    
    env.close()
  
  
  
