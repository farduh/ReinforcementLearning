#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:42:19 2022

@author: farduh
"""



import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from q_learning import FeatureTransformer,plot_cost_to_go,plot_running_avg

# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False

# Inspired by https://github.com/dennybritz/reinforcement-learning

class BaseModel():
    
    def __init__(self,D):
        self.w = np.random.random(D)/np.sqrt(D)
        
        
    def partial_fit(self,X,Y,eligibility,lr=1e-2):
        self.w += lr*(Y-X.dot(self.w))*eligibility
        
    def predict(self,X):
        X = np.array(X)
        return X.dot(self.w)
            
# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer ):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    D = feature_transformer.dimensions
    self.eligibilities = np.zeros((env.action_space.n,D))
    for i in range(env.action_space.n):
      model = BaseModel(D)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    result = np.array([m.predict(X)[0] for m in self.models])
    return result

  def update(self, s, a, G,gamma,lambda_):
    X = self.feature_transformer.transform([s])
    self.eligibilities *= gamma*lambda_
    self.eligibilities[a] += X[0]
    assert(len(X.shape) == 2)
    self.models[a].partial_fit(X[0], [G],self.eligibilities[a])

  def sample_action(self, s, eps):
    # eps = 0
    # Technically, we don't need to do epsilon-greedy
    # because SGDRegressor predicts 0 for all states
    # until they are updated. This works as the
    # "Optimistic Initial Values" method, since all
    # the rewards for Mountain Car are -1.
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma, lambda_):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 10000:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # update the model
    next = model.predict(observation)
    # assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next[0])
    model.update(prev_observation, action, G, gamma,lambda_)

    totalreward += reward
    iters += 1

  return totalreward


def main(show_plots=True):
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft)
  gamma = 0.99
  lambda_ = 0.7
  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 10000
  totalrewards = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**n)
    if n == 199:
      print("eps:", eps)
    # eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(model, env, eps, gamma,lambda_)
    totalrewards[n] = totalreward
    if (n + 1) % 100 == 0:
      print("episode:", n, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())

  if show_plots:
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)


if __name__ == '__main__':
  # for i in range(10):
  #   main(show_plots=False)
  main()