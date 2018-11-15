import numpy as np

class MAB:
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

    
# To create the environment use:
number_bandits = 10
mab_environment = MAB(num_b = number_bandits)

# To do an action over the environment and get the reward use:
action = 1 # As we have 10 bandits, action must be a number between 0 and 9
reward = mab_environment.get_reward(action)
