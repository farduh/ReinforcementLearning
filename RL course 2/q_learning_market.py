#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:25:54 2022

@author: farduh
"""

#import quantstats as qs
import sys
sys.path.append('/home/farduh/xcapit/xcapit_util')
from candlestick import BinancePricesRepository
from datetime import datetime
import pandas as pd
import numpy as np
from q_learning import plot_cost_to_go,plot_running_avg
import gym
from gym import spaces
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
from tqdm import tqdm 
from trading_environment import StockTradingEnv
from ta.volatility import bollinger_hband,bollinger_lband
from ta.momentum import rsi
import pickle
import theano 
import theano.tensor as T

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

    observation_examples = np.random.random((20000,env.dimensions*env.n_observation+1))
    observation_examples[:,env.dimensions*env.n_observation] = np.round(observation_examples[:,env.dimensions*env.n_observation])
    observation_examples[:,:env.dimensions*env.n_observation] = observation_examples[:,:env.dimensions*env.n_observation]/10.-1
    observation_examples 
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=2., n_components=n_components)),
            ("rbf2", RBFSampler(gamma=1., n_components=n_components)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=n_components))
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


class BaseModel():
    
    def __init__(self,D,gamma,lambda_,lr=1e-1):
        self.w = np.random.random(D)/np.sqrt(D)
        self.b = np.random.random(D)/np.sqrt(D)
        self.eligibility_w = np.zeros(D)
        self.eligibility_b = np.zeros(D)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.steps = 0
        
        # thX = T.vector('X')
        # thT = T.scalar('T')
        # w = theano.shared(np.random.random(D)/np.sqrt(D), 'w')
        # b = theano.shared(np.random.random(1), 'b')
        # eligibility_w = theano.shared(np.zeros(D), 'eligibility_w')
        # eligibility_b = theano.shared(np.zeros(1), 'eligibility_b')
        # thY = thX.dot(w)+b
        # cost = (thT-thY).dot(thT-thY)
        # update_eligibility_w = eligibility_w*gamma*lambda_+ thX
        # update_eligibility_b = eligibility_b*gamma*lambda_+ 1
        # update_w = w - lr*T.grad(cost,w)*eligibility_w
        # update_b = b - lr*T.grad(cost,b)*eligibility_b
        
        # self.train = theano.function(
        #     inputs=[thX,thT],
        #     updates=[(w,update_w),(b,update_b),
        #               (eligibility_w,update_eligibility_w),
        #               (eligibility_b,update_eligibility_b)])

        # self.get_prediction = theano.function(
        #     inputs=[thX],
        #     outputs = thY)
        
        
    def partial_fit(self,X,Y,lr=1e-1):
        self.eligibility_w *= self.gamma*self.lambda_
        self.eligibility_w += X
        self.eligibility_b *= self.gamma*self.lambda_
        self.eligibility_b += 1
        self.w += lr*(Y-X.dot(self.w)-self.b)*self.eligibility_w
        self.b += lr*(Y-X.dot(self.w)-self.b)*self.eligibility_b
        self.steps += 1
        # self.train(X,Y)

        
    def predict(self,X):
        X = np.array(X)
        return X.dot(self.w)+self.b
        #return self.get_prediction(X)
    
# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer,gamma,lambda_ ):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    D = feature_transformer.dimensions
    for i in range(env.action_space.n):
      #model = BaseModel(D,gamma,lambda_)
      X = self.feature_transformer.transform([env.reset()])
      print("eta0=0.05,learning_rate='constant',penalty='elasticnet',l1_ratio=0.5")
      model = SGDRegressor(eta0=0.05,learning_rate='constant',penalty='elasticnet',
                           l1_ratio=0.5)
      model.partial_fit(X,[0])
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    result = np.array([m.predict(X)[0] for m in self.models])
    return result

  def update(self, s, a, G,gamma,lambda_):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):

    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma, lambda_):
  observation = env.reset()
  done = False
  totalreward = 1
  iters = 0
  rewards = []

  while not done :
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # update the model
    next = model.predict(observation)
    # assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next[0])
    model.update(prev_observation, action, G, gamma,lambda_)
    
    rewards.append(reward)
    totalreward *= (1+reward)
    iters += 1

  return totalreward,rewards

def play_one_test(model, env, eps, gamma,lambda_,update=True):
  observation = env.reset()
  done = False
  totalreward = 1
  iters = 0
  rewards = []
  while not done:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # update the model
    next = model.predict(observation)
    # assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next[0])
    if update:
        model.update(prev_observation, action, G, gamma,lambda_)
    rewards.append(reward)
    totalreward *= (1+reward)
    iters += 1

  return totalreward,rewards

def save_model(model,filepath='model_weights/'):
    
    pickle.dump( model.models[0].w, open( filepath+"basic_model_w_0.p", "wb" ) ) 
    pickle.dump( model.models[0].b, open( filepath+"basic_model_b_0.p", "wb" ) ) 
    pickle.dump( model.models[1].w, open( filepath+"basic_model_w_1.p", "wb" ) ) 
    pickle.dump( model.models[1].b, open( filepath+"basic_model_b_1.p", "wb" ) ) 
    pickle.dump( model.feature_transformer.featurizer,\
                open( filepath+"featurizer.p", "wb" ) ) 
    pickle.dump( model.feature_transformer.scaler,\
               open( filepath+"scaler.p", "wb" ) ) 
  
    
def load_model(model,filepath='model_weights/'):
     
    model.models[0].w = pickle.load( open( filepath+"basic_model_w_0.p", "rb" ) ) 
    model.models[0].b = pickle.load( open( filepath+"basic_model_b_0.p", "rb" ) ) 
    model.models[1].w = pickle.load( open( filepath+"basic_model_w_1.p", "rb" ) ) 
    model.models[1].b = pickle.load( open( filepath+"basic_model_b_1.p", "rb" ) ) 
    model.feature_transformer.featurizer = \
        pickle.load(open( filepath+"featurizer.p", "rb" ) ) 
    model.feature_transformer.scaler = \
        pickle.load(open( filepath+"scaler.p", "rb" ) ) 
  
    return model

def preprocess_candles(prices,derivatives=0):
    
    sma_10 = prices['close'].rolling(10).mean()
    sma_55 = prices['close'].rolling(55).mean()
    vsma_10 = prices['volume'].rolling(10).mean()
    vsma_55 = prices['volume'].rolling(55).mean()
    hbb = bollinger_hband(prices['close'],window=21)
    lbb = bollinger_lband(prices['close'],window=21)
    pct_h = prices['high'].rolling(10).mean()/np.maximum(prices['close'],prices['open']).rolling(10).mean()
    pct_l = np.minimum(prices['close'],prices['open']).rolling(10).mean()/prices['low'].rolling(10).mean()
    
    candles = pd.DataFrame()
    candles['close_sma'] = sma_10.multiply(1/sma_55)
    candles['volume_sma'] = vsma_10.multiply(1/vsma_55)
    candles['close_std'] = (prices['close']-lbb).multiply(1/(hbb-lbb))+0.5
    candles['close_rsi'] = rsi(prices['close'])/100.+0.5
    candles['pct_h'] = pct_h
    candles['pct_l'] = pct_l

    if derivatives > 0 :
        pct_candles = candles/candles.shift(1)
        new_names = { x : x+'_pct' for x in pct_candles.columns}
        pct_candles = pct_candles.rename(columns=new_names)
        candles = candles.join(pct_candles)
    elif derivatives > 1 :
        pct_candles = candles.diff()
        new_names = { x : x+'_pct2' for x in pct_candles.columns}
        pct_candles = pct_candles.rename(columns=new_names)
        candles = candles.join(pct_candles)
    
    candles = candles.dropna()

    return candles

def main(show_plots=True):
    
    prices = BinancePricesRepository().get_candles('BTC', 'USDT', '4h', 10000,datetime(2022,5,1))
    prices = prices.drop(columns='close_time')
    candles = preprocess_candles(prices,derivatives=0)
    train_candles = candles.iloc[:int(0.8*len(prices))]
    test_candles = candles.iloc[int(0.8*len(prices)):]
    n_observation = 4
    env = StockTradingEnv(train_candles,prices,n_observation)
    ft = FeatureTransformer(env)
    gamma = 0.99
    lambda_ = 0.7
    model = Model(env,ft,gamma,lambda_)
    
    if 'monitor' in sys.argv:
      filename = os.path.basename(__file__).split('.')[0]
      monitor_dir = './' + filename + '_' + str(datetime.now())
      env = wrappers.Monitor(env, monitor_dir)
      
          
    N = 200
    totalrewards = np.empty(N)
    for n in range(N):
      #eps = 0.1/(0.1*n+1)
      eps = 0.1*(0.97**n)
      #eps = 1./np.sqrt(n+1)

      # eps = 1.0/np.sqrt(n+1)
      totalreward, rewards = play_one(model, env, eps, gamma,lambda_)
      totalrewards[n] = totalreward
      #if ((n % 10 == 0 and n<100) or (n % 100) and n!=0:
      print("\r episode:", n, "total reward:", totalreward,"eps:", eps)
        # df = pd.DataFrame(index=env.candles.index[n_observation-1:])
        # df['rewards'] = np.append(np.zeros(1),rewards)
        # (1+df['rewards']).cumprod().plot()
        # (1+env.prices['close'].pct_change()).cumprod().plot()
        # plt.show()
    # save_model(model,filepath='model_weights_2/')
    # pickle.dump( totalrewards, open( "total_rewards.p", "wb" ) ) 
    
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())
    
    
    
    if show_plots:
      plt.figure(figsize=(12,6))
    
      plt.plot(totalrewards,label='Performance per episode')
      plt.plot(pd.Series(totalrewards).rolling(100).mean(),label='Average Performance in 100 episodes')
      
      #plt.title("Rewards")
      plt.xlabel("episodes")
      plt.title("Cumulative product of (1+Reward)")
      plt.legend()
    #  plt.show()
    
      plot_running_avg(totalrewards)

    # model = load_model(model,filepath='model_weights_2/')
    env = StockTradingEnv(train_candles,prices,n_observation)
    totalreward, rewards = play_one(model, env, 0, gamma,lambda_)
    plt.figure(figsize=(12,6))
    df = pd.DataFrame(index=env.candles.index)
    df['rewards'] = np.append(np.zeros(n_observation),rewards)
    (1+env.prices['close'].pct_change().loc[train_candles.index]).cumprod().plot(label='BTC/USDT',c='k')
    (1+df['rewards']).cumprod().plot(label='Agent',c='r')
    plt.ylabel('performance')
    plt.title(f'Performance with train data: {(1+pd.Series(rewards)).cumprod().iloc[-1]:.2f}')
    plt.legend()
    plt.show()
    
    env = StockTradingEnv(test_candles,prices,n_observation)
    totalreward, rewards = play_one(model, env, 0, gamma,lambda_)   
    print("test total reward:", totalreward)
    plt.figure(figsize=(12,6))
    df = pd.DataFrame(index=env.candles.index)
    df['rewards'] = np.append(np.zeros(n_observation),rewards)
    (1+env.prices['close'].pct_change().loc[test_candles.index]).cumprod().plot(label='BTC/USDT',c='k')
    (1+df['rewards']).cumprod().plot(label='Agent',c='r')
    plt.ylabel('performance')
    plt.title(f'Performance with test data: {(1+pd.Series(rewards)).cumprod().iloc[-1]:.2f}')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
  # for i in range(10):
  #   main(show_plots=False)
  main()

    