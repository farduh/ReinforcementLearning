#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:26:02 2022

@author: farduh
"""

import gym
from gym import spaces
import pandas as pd
import numpy as np
from ta.volatility import bollinger_hband,bollinger_lband
from ta.momentum import rsi

class StockTradingEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, candles,prices,n_observation=1):
    super(StockTradingEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    
    self.dimensions = len(candles.columns)
    self.n_observation = n_observation
    self.prices = prices
    self.candles = candles
    self.observation = None
    self.current_index = None
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(2)
    lower_bound = [-1 for x in range(self.dimensions*self.n_observation)]+[-1]
    higher_bound = [ 2 for x in range(self.dimensions*self.n_observation)]+[2]
    self.observation_space = spaces.Box(low=np.array(lower_bound), high=np.array(higher_bound), shape=
                    (self.dimensions*self.n_observation+1, ), dtype=np.float32)

  def step(self, action):
    # Execute one time step within the environment
    self.current_index += 1
    last_state = self.observation[-1]
    next_state = 2
    reward = 0
    if last_state == 1:
        if action == last_state:
            next_state = 1
        else:
            next_state = 0
            reward -= 0.001
    if last_state == 0:
        if action == last_state:
            next_state = 0
        else:
            next_state = 1
            reward -= 0.001
    
    assert next_state<2
    
    self.observation = np.append(self.observation[self.dimensions:self.dimensions*(self.n_observation)],
                                 np.append(self.candles.iloc[self.current_index],next_state))
    done = False
    if self.candles.index[self.current_index] == self.candles.index[-1]:
        done = True
    index_date = self.candles.index[self.current_index]
    if next_state == 1:
        reward += self.prices.loc[index_date,'close']/self.prices.loc[index_date,'open']-1
    # if next_state == 0:
    #         reward += self.prices.loc[index_date,'open']/self.prices.loc[index_date,'close']-1
    return self.observation, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    self.observation = np.array([])
    self.current_index = -1
    for _ in range(self.n_observation):
        self.current_index += 1
        self.observation = np.append(self.observation,self.candles.iloc[self.current_index])
        
    self.observation = np.append(self.observation,0)
    
    return self.observation

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return None
