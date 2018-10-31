import numpy as np

class MBA:
  ''' Creates a multiarmed bandit where each bandit outputs a reward coming 
  from a Gaussian distribution. '''
  
  def __init__(self, num_b):
    
    self.N = num_b # Number of bandits
    self.prob_b = np.random.normal(0, 1, self.N) # Reward probability of each bandit
    
          
  def get_reward(self, action):
    # Given action, outputs a reward based on the probability of success of bandit
    if action > self.N-1:
      print('Bandit number'+str(action)+' does not exists!')
      return 0
    else: 
      reward = np.random.normal(self.prob_b[action],1)
      return reward
