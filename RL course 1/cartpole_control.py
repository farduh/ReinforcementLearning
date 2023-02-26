import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from iterative_policy_evaluation_deterministic import print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler

GAMMA = 0.99
ALPHA = 0.1

def epsilon_greedy(model, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    values = model.predict_all_actions(s)
    return np.argmax(values)
  else:
    return model.env.action_space.sample()


def gather_samples(env, n_episodes=100000):
  samples = []
  for _ in range(n_episodes):
    s = env.reset()
    done = False
    while not done:
      a = env.action_space.sample()
      sa = np.concatenate((s, [a]))
      samples.append(sa)
      s, r, done,info = env.step(a)

  return samples


class Model:
  def __init__(self, env):
    # fit the featurizer to data
    self.env = env
    samples = gather_samples(env)
   # self.featurizer = Nystroem()
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)
    dims = self.featurizer.n_components

    # initialize linear model weights
    self.w = np.zeros(dims)

  def predict(self, s, a):
    sa = np.concatenate((s, [a]))
    x = self.featurizer.transform([sa])[0]
    return x @ self.w

  def predict_all_actions(self, s):
    return [self.predict(s, a) for a in range(self.env.action_space.n)]

  def grad(self, s, a):
    sa = np.concatenate((s, [a]))
    x = self.featurizer.transform([sa])[0]
    return x


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  # grid = standard_grid()
  env = gym.make('CartPole-v0')
  
  # print rewards
  
  model = Model(env)
  reward_per_episode = []
  #state_visit_count = {}

  # repeat until convergence
  n_episodes = 1500
  for it in range(n_episodes):
    
    s = env.reset()
    #state_visit_count[s] = state_visit_count.get(s, 0) + 1
    episode_reward = 0
    done = False
    while not done:
      a = epsilon_greedy(model, s)
      s2, r, done,info = env.step(a)
      #state_visit_count[s2] = state_visit_count.get(s2, 0) + 1

      # get the target
      if done:
        target = r
      else:
        values = model.predict_all_actions(s2)
        target = r + GAMMA * np.max(values)

      # update the model
      g = model.grad(s, a)
      err = target - model.predict(s, a)
      model.w += ALPHA * err * g
      
      # accumulate reward
      episode_reward += r

      # update state
      s = s2
    
    reward_per_episode.append(episode_reward)
    
    if (it + 1) % 50 == 0:
      print(f"Episode: {it + 1}, Reward: {episode_reward}")

    # early exit
    if it > 20 and np.mean(reward_per_episode[-20:]) == 200:
      print("Early exit")
      break

  plt.plot(reward_per_episode)
  plt.title("Reward per episode")
  plt.show()
