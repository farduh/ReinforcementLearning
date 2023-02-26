# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:40:32 2022

@author: Fran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
import gym
import argparse
from iterative_policy_evaluation_deterministic import print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import StandardScaler
import os
import pickle
from tqdm import tqdm

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def get_data():
    
    #df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/aapl_msi_sbux.csv')
    df = pd.read_csv('BTCUSDT_1d.csv')*100
     
    return df.values


def epsilon_greedy(model, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    values = model.predict_all_actions(s)
    return np.argmax(values)
  else:
    return model.env.action_space


def gather_states(env):
  states = []
  for _ in tqdm(range(env.n_step)):
    s = env.reset()
    done = False
    while not done:
      a = np.random.choice(env.action_space)
      #sa = np.concatenate((s, [a]))
      states.append(s)
      s, r, done,info = env.step(a)

  return states

def get_scaler(env):
    states = gather_states(env)
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

class LinearModel:
    def __init__(self, input_dim,n_action):
      # fit the featurizer to data
      
      self.W = np.random.randn(input_dim,n_action)/np.sqrt(n_action)
      self.b = np.zeros(n_action)
      
      #momentum terms
      self.vW = 0
      self.vb = 0
      
      self.losses = []
    
    def predict(self, X):
      assert(len(X.shape)==2)
      return X @ self.W + self.b
    
    def predict_all_actions(self, s):
      return [self.predict(s, a) for a in range(self.env.action_space.n)]
    
    def sgd(self, X, Y, learning_rate = 0.01, momentum = 0.9):
      assert(len(X.shape)==2)
      num_values = np.prod(Y.shape)
      Yhat = self.predict(X)
      gW = 2*X.T.dot(Yhat - Y)/num_values
      gb = 2*(Yhat - Y).sum(axis=0)/num_values
      
      self.vW = momentum * self.vW - learning_rate * gW
      self.vb = momentum * self.vb - learning_rate * gb
      
      self.W += self.vW
      self.b += self.vb
      
      mse = np.mean((Yhat-Y)**2)
      self.losses.append(mse)

    def load_weights(self,filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']
        
    def save_weights(self,filepath):
        np.savez(filepath,W=self.W,b=self.b)

class MultiStockEnv:
    
    def __init__(self,stock_prices,initial_investment):
        
        #data
        self.stock_prices_history = stock_prices
        self.n_step,self.n_stock = self.stock_prices_history.shape
        # instance attribute
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.action_space = np.arange(3**self.n_stock)
        self.action_list = list(map(list,itertools.product([0,1,2],repeat=self.n_stock)))
        self.state_dim = 2*self.n_stock + 1
        self.reset()
        
    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_prices_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()
    
    def step(self,action):
        assert action in self.action_space
        #get current value before performing the action
        prev_val = self._get_val()
        
        #update price, ie go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_prices_history[self.cur_step]
        
        #perform the trade
        self._trade(action)
        
        #get the new value after taking  the action
        cur_val = self._get_val()
        
        #reward is the increase of portfolio value
        reward = cur_val - prev_val
        
        #done if we have run of data
        done = (self.cur_step == (self.n_step - 1))
        
        #store current value of the portfolio here
        info = {'portfolio_value':cur_val}
        
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        if self.cur_step != 0:
            obs[self.n_stock:self.n_stock*2] = self.stock_price/self.stock_prices_history[self.cur_step-1]
        else:
            obs[self.n_stock:self.n_stock*2] = self.stock_price/self.stock_price
        obs[-1] = self.cash_in_hand
        return obs
    
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
    
    def _trade(self,action):
        #index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        action_vec = self.action_list[action]
        
        #determine which stock to buy or sell
        sell_index = []
        buy_index = []
        for i,a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)
                
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_owned[i]*self.stock_price[i]
                self.stock_owned[i] = 0 
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False
                    
                

class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)
    
    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    
    def train(self,state,action,reward,next_state,done):
        # get the target
        if done:
          target = reward
        else:
          values = self.model.predict(next_state)
          target = reward + self.gamma * np.amax(values,axis=1)
        
        target_full = self.model.predict(state)
        target_full[0,action] = target
        self.model.sgd(state,target_full)
            
        if self.epsilon <= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)
        
    def save(self,name):
        self.model.save_weights(name)
        

def play_one_episode(agent,env,train_mode=True):
    state = env.reset()
    state = scaler.transform([state])
    done = False
    invest_evolution = []
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        #if not train_mode:
        invest_evolution.append(info['portfolio_value'])
        next_state = scaler.transform([next_state])
        if train_mode:
            agent.train(state,action,reward,next_state,done)
        state = next_state
        
    return info['portfolio_value'],invest_evolution


if __name__ == '__main__':
    
    #config
    models_folder = 'linear_rl_trader_models_1'
    rewards_folder = 'linear_rl_trader_rewards_1'
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data()
    n_timesteps,n_stock = data.shape
    
    n_train = n_timesteps//2
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    env = MultiStockEnv(data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    
    #store the final value of the portfolio (end of episode)
    portfolio_value = []
    
    train_mode = (args.mode == 'train')
    
    if not train_mode:
      # then load the previous scaler
      with open(f'{models_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
      
      # remake the env with test data
      env = MultiStockEnv(test_data, initial_investment)
      
      # make sure epsilon is not 1!
      # no need to run multiple episodes if epsilon = 0, it's deterministic
      agent.epsilon = 0.01
      
      # load trained weights
      agent.load(f'{models_folder}/linear.npz')
     

    # play the game num_episodes times
    for e in range(num_episodes):
      t0 = datetime.now()
      val ,_ = play_one_episode(agent, env, train_mode)
      dt = datetime.now() - t0
      print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
      portfolio_value.append(val) # append episode end portfolio value
      
    # save the weights when we are done
    if train_mode:
      # save the DQN
      agent.save(f'{models_folder}/linear.npz')
      
      # save the scaler
      with open(f'{models_folder}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
      
      # plot losses
      plt.plot(agent.model.losses)
      plt.show()
      
      
    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

    if not train_mode:
        agent.epsilon = 0.0
        t0 = datetime.now()
        val, invest_evolution = play_one_episode(agent, env, train_mode=True)
        df = pd.DataFrame()
        for i in range(test_data.shape[1]):
            df[str(i)] = test_data[:,i]
        df['portfolio'] = [initial_investment]+invest_evolution
        df.to_csv(f'{rewards_folder}/portfolio_evolution.csv',index=False)
