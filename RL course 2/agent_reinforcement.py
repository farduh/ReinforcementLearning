#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:03:26 2022

@author: farduh
"""


#import quantstats as qs
import sys
sys.path.append('/home/farduh/xcapit/xcapit_util')
from candlestick import BinancePricesRepository
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from q_learning import plot_cost_to_go,plot_running_avg
import gym
from gym import spaces
import os
import sys
from tqdm import tqdm 
from ta.volatility import bollinger_hband,bollinger_lband
from ta.momentum import rsi
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import seaborn as sns


class StockTradingEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, prices,n_observation=1):
    super(StockTradingEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    candles = pd.DataFrame()
    
    sma_10 = prices['close'].rolling(10).mean()
    sma_55 = prices['close'].rolling(55).mean()
    vsma_10 = prices['volume'].rolling(10).mean()
    vsma_55 = prices['volume'].rolling(55).mean()
    hbb = bollinger_hband(prices['close'],window=21)
    lbb = bollinger_lband(prices['close'],window=21)
    pct_h = prices['high'].rolling(10).mean()/np.maximum(prices['close'],prices['open']).rolling(10).mean()
    pct_l = np.minimum(prices['close'],prices['open']).rolling(10).mean()/prices['low'].rolling(10).mean()
    
    candles['close_sma'] = sma_10.multiply(1/sma_55)
    candles['volume_sma'] = vsma_10.multiply(1/vsma_55)
    candles['close_std'] = (prices['close']-lbb).multiply(1/(hbb-lbb))
    candles['close_rsi'] = rsi(prices['close'])/100.
    candles['pct_h'] = pct_h
    candles['pct_l'] = pct_l
    
    self.dimensions = len(candles.columns)
    self.scaler = StandardScaler()
    self.n_observation = n_observation
    self.prices = prices
    self.candles = pd.DataFrame(self.scaler.fit_transform(candles),index = prices.index)
    self.candles = candles.dropna()
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
    return self.observation, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    self.observation = np.append(np.ones(self.dimensions*(self.n_observation-1)),np.append(self.candles.iloc[0],0))
    self.current_index = 0
    return self.observation

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return None



def main(show_plots=True):
    
  prices = BinancePricesRepository().get_candles('BTC', 'USDT', '4h', 10000,datetime(2022,5,1))
  prices = prices.drop(columns='close_time')
  env = StockTradingEnv(prices.iloc[:int(0.8*len(prices))],n_observation = 4)
  model = PPO('MlpPolicy',DummyVecEnv([lambda:  env]), n_steps=8, 
              learning_rate=1e-3,ent_coef=0.001, clip_range=0.1,gae_lambda=0.7)#, nminibatches=1, noptepochs=10, verbose=2, ent_coef=0.001, cliprange=0.1)
  model.learn(total_timesteps=10000)
  
  
  
  #env = StockTradingEnv(prices.iloc[int(0.8*len(prices)):],n_observation = 4)
  obs = env.reset()
  total_return = []
  benchmark_return = []
  actions = []
  returns = 1
  benchmark = 1
  done = False
  while not done:
      action, _states = model.predict(obs)
      obs, rewards, done, info = env.step(action)
      returns *= (1+rewards)
      total_return.append(returns)
      index_date = env.candles.index[env.current_index]
      benchmark *= (env.prices.loc[index_date,'close']/env.prices.loc[index_date,'open'])
      benchmark_return.append(benchmark)
      actions.append(action)
  plt.plot(total_return)
  plt.plot(benchmark_return)
  df= pd.DataFrame(np.array([total_return,benchmark_return,actions]).T)
  

