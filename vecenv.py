import numpy as np

class VecEnv():
  def __init__(self, makeEnv, nenvs):
    self.envs = []
    for i in range(nenvs):
      self.envs.append(makeEnv())

    self.states = []
    self.nenvs = nenvs
    self.cumulative_rewards = [0 for i in range(nenvs)]

    states = []
    for e in self.envs:
      states.append(e.reset())
    self.states = np.stack(states)

  ''' returns current state of envs '''
  def getStates(self):
    return self.states


  ''' returns stacked states, rewards, and dones for the environments
  after taking one step. Also returns a list of episode_scores for the total score of
  any episodes that terminated. '''
  def step(self, actions):
    rewards = []
    dones = []

    episode_scores = []

    for i,env in enumerate(self.envs):
      self.states[i], reward, done, info = env.step(actions[i])

      self.cumulative_rewards[i] += reward
      rewards.append(reward)
      dones.append(done)

      if done:
        self.states[i] = env.reset()
        episode_scores.append(self.cumulative_rewards[i])
        self.cumulative_rewards[i] = 0
        
    return self.states, np.stack(rewards), np.stack(dones), episode_scores
      
