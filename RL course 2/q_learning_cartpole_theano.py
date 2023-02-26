#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:37:08 2022

@author: farduh
"""

#from q_learning_cartpole_linear import *
import theano as th
import theano.tensor as T

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
from tensorflow import keras
import tensorflow as tf
# class SGDRegressor():
    
#     def __init__(self,D,eta0=10e-2,**kwargs):
#         print('Hello Theano!')
#         W_init = np.random.random(D)/np.sqrt(D)
#         thX = T.matrix('X')
#         thT = T.vector('T')
#         W = th.shared(W_init, 'W')
#         thY = thX.dot(W)
#         cost = (thY-thT).dot(thY-thT)
#         update_W = W - eta0*T.grad(cost,W)
        
#         self.train = th.function(
#             inputs=[thX,thT],
#             updates=[(W,update_W)])

#         self.get_prediction = th.function(
#             inputs=[thX],
#             outputs = thY)
        
#     def partial_fit(self,X,Y):
#         self.train(X,Y)
        
#     def predict(self,X):
#         return self.get_prediction(X)


class SGDRegressor():
    
    def __init__(self,D,eta0=10e-2,**kwargs):
        print('Hello keras!')
        inputs = keras.Input(shape=(D,), name="digits")
        outputs = keras.layers.Dense(1, activation='linear',name="predictions")(inputs)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = keras.optimizers.SGD(
            learning_rate=eta0, momentum=0.0, nesterov=False, name="SGD")
        self.loss_fn = keras.losses.MeanSquaredError()
        
        # self.model.compile(loss='mse',
        #       optimizer=optimizer,
        #       metrics=['mae', 'mse'])
        
    def partial_fit(self,X,Y):
        # Iterate over the batches of the dataset.
        with tf.GradientTape() as tape:
            Yhat = self.model(X, training=True)
            loss_value = self.loss_fn(Y, Yhat)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    def predict(self,X):
        return self.model.predict(X)[0]

class FeatureTransformer:
    
  def gather_samples(self,env,episodes=10000):
      observations = []
      for _ in range(episodes):
          observations.append(env.reset())
          done = False
          iters = 0
          while not done and iters < 10000:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            observations.append(observation)
      return np.array(observations)
    
    
  def __init__(self, env, n_components=500):
    #observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    #observation_examples = self.gather_samples(env)
    observation_examples = np.random.random((20000,4))*2-2
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    # print "observations:", observations
    scaled = self.scaler.transform(observations)
    # assert(len(scaled.shape) == 2)
    return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      model = SGDRegressor(feature_transformer.dimensions,eta0=0.1,learning_rate='constant')
      #model.partial_fit(feature_transformer.transform( [env.reset()] ), [0])
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    result = np.stack([m.predict(X) for m in self.models]).T
    
    assert(len(result.shape) == 2)
    return result

  def update(self, s, a, G):
    X = self.feature_transformer.transform([s])
    #assert(len(X.shape) != 2)
    self.models[a].partial_fit(X, np.array([G]))

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
def play_one(model, env, eps, gamma):
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
    model.update(prev_observation, action, G)

    totalreward += reward
    iters += 1
    
  return totalreward


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def main(show_plots=True):
  env = gym.make('CartPole-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft, "constant")
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 1000
  totalrewards = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**n)
    if n == 199:
      print("eps:", eps)
    # eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(model, env, eps, gamma)
    totalrewards[n] = totalreward
    if (n ) % 10 == 0:
      print("episode:", n, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())

  if show_plots:
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)




if __name__ == '__main__':
  # for i in range(10):
  #   main(show_plots=False)
  main()
        