# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:25:59 2022

@author: Fran
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

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


def gather_samples(env, n_episodes=10000):
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
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    
   # self.featurizer = Nystroem()
    n_components = 500
    self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    self.scaler = StandardScaler()
    scale_samples = self.scaler.fit_transform(observation_examples)
    self.featurizer.fit(scale_samples)
    self.regressors = []
    # initialize linear model weights
    for i in range(env.action_space.n):
        regressor = SGDRegressor()
        s = env.reset()
        regressor.partial_fit(self.featurizer.transform(self.scaler.transform([s])),[0])
        self.regressors.append(regressor)
        
  def predict(self, s, a):
    x = self.scaler.transform([s])[0]
    x = self.featurizer.transform([x])[0]
    return self.regressors[a].predict(x.reshape(1,-1))

  def predict_all_actions(self, s):
    return [self.predict(s, a) for a in range(self.env.action_space.n)]

  def update(self, s, a,target):
    x = self.scaler.transform([s])[0]
    x = self.featurizer.transform([x])[0]
    self.regressors[a].partial_fit([x], [target])
    return 


if __name__ == '__main__':
  # instantiate environment
  env = gym.make("MountainCar-v0")

  model = Model(env)
  reward_per_episode = []

  # watch untrained agent
  #watch_agent(model, env, eps=0)

  # repeat until convergence
  n_episodes = 500
  for it in range(n_episodes):
    eps = 0.01*(0.97**it)
    s = env.reset()
    episode_reward = 0
    done = False
    while not done:
      a = epsilon_greedy(model, s,eps=eps)
      s2, r, done, info = env.step(a)
      # get the target
      if done:
        target = r
      else:
        values = model.predict_all_actions(s2)
        target = r + GAMMA * np.max(values)

      # update the model
      model.update(s, a,target)
      
      # accumulate reward
      episode_reward += r

      # update state
      s = s2
    if (it + 1) % 50 == 0:
      print(f"Episode: {it + 1}, Reward: {episode_reward}")

    # early exit
    if it > 20 and np.mean(reward_per_episode[-20:]) > 0:
      print("Early exit")
      break
    
    reward_per_episode.append(episode_reward)

  # test trained agent
  #test_reward = test_agent(model, env)
  #print(f"Average test reward: {test_reward}")

  plt.plot(pd.Series(reward_per_episode).rolling(30).mean())
  plt.title("Reward per episode")
  plt.show()

matrix = []
for y in np.linspace(-0.8, 0.8, 100):
    line = []
    for x in np.linspace(-1.5, 1.5, 100):    
        line.append(np.sum(model.predict_all_actions([x,y])))
        
    matrix.append((line))

import seaborn as sns
sns.heatmap(matrix)
        
    
s = env.reset()
episode_reward = 0
done = False
eps = 0 
while not done:
  a = epsilon_greedy(model, s,eps=eps)
  s2, r, done, info = env.step(a)
  
  if s2[0]>0:
      r += s2[0]
  # get the target
  if done:
    target = r
  else:
    values = model.predict_all_actions(s2)
    target = r + GAMMA * np.max(values)

  # update state
  s = s2
  # accumulate reward
  episode_reward += r

  env.render()
print(episode_reward)
env.close()
env = gym.make("MountainCar-v0")
