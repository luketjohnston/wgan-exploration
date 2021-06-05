import numpy as np
import code

class ReplayBuffer():
  '''
  simple replay buffer
  '''
  def __init__(self, n, data_shape, dtype):
    self.n = n
    self.buffer = np.zeros(shape=[n,]  + list(data_shape), dtype=dtype)
    self.i = 0
    self.num_datapoints = 0
    self.data_shape = data_shape

  def isFull(self):
    return self.num_datapoints == self.n

  
  def add(self, data):
    assert(list(data.shape[1:]) == list(self.data_shape)), str(data.shape[1:]) + ' is not equal to ' + str(self.data_shape)
    num_new = data.shape[0]
    indices = [x % self.n for x in range(self.i, self.i + num_new)]
    self.buffer[indices, ...] = data

    self.i = (self.i + num_new) % self.n
    self.num_datapoints = min(self.n, self.num_datapoints + num_new)

  def sample(self, n):
    indices = np.random.randint(self.num_datapoints, size=[n])
    return np.take(self.buffer, indices, axis=0, mode="wrap")

  ''' 
  note: doesn't save self.i or self.num_datapoints. 
  loads with self.i = 0 and self.num_datapoints = self.n
  '''
  def saveToFile(self, filename):
    assert self.isFull()
    with open(filename, "wb") as fp:
      np.save(fp, self.buffer)

  def loadFromFile(self, filename):
    with open(filename, "rb") as fp:
      self.buffer = np.load(fp)
      self.num_datapoints = self.n
      assert self.buffer.shape[0] == self.n
    
    
