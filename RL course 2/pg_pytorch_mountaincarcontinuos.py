#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:13:57 2022

@author: farduh
"""


import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Policy(nn.Module):
    def __init__(self, s_size,  h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc_mean = nn.Linear(h_size, 1)
        self.fc_std = nn.Linear(h_size, 1)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        return torch.tanh(mean),F.softplus(std)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean,std = self.forward(state)
        if torch.isnan(mean):
            return np.random.uniform(),0
        mean.cpu()
        std.cpu()
        m = torch.distributions.Normal(mean,std)
        action = m.sample()
        one = torch.tensor(1)
        action = max(-1*one,min(one,action))
        return action.item(), m.log_prob(action)
    
    def partial_fit(self,loss):
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class Value(nn.Module):
    def __init__(self, s_size,  h_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc_value = nn.Linear(h_size, 1)    

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc_value(x)
        return value
    
    def value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = self.forward(state).cpu()
        return value
    
    def partial_fit(self,pred,y):
        # Backpropagation
        
        loss = F.mse_loss(pred, torch.tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

learning_rate = 1e-3
gamma = 0.99

env = gym.make('MountainCarContinuous-v0')
D = env.observation_space.shape[0]
#K = env.action_space.n
#pmodel = PolicyModel(D, K, [16,2],None,None)

pmodel = Policy(D,  32)
vmodel = Value(D,  32)

poptimizer = torch.optim.Adam(pmodel.parameters(), lr=learning_rate)
pmodel.optimizer = poptimizer
voptimizer = torch.optim.Adam(vmodel.parameters(), lr=learning_rate)
vmodel.optimizer = voptimizer


totalrewards = []
losses = []
policy_losses = []

N = 200
for n in range(N):
    reward = 0
    totalreward = 0
    actions = []
    rewards = []
    states = []
    log_probs = []
    observation = env.reset()
    done = False
    while not done:
        action, log_prob = pmodel.act(observation)#torch.tensor(observation).reshape(1,1,D))
        log_probs.append(log_prob)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        prev_observation = observation
        observation, reward, done, info = env.step([action])
   
        #if reward == 1: # if we changed the reward to -200
        totalreward += reward
    totalrewards.append(totalreward)
    action, log_prob = pmodel.act(observation)#torch.tensor(observation).reshape(1,1,D))
    states.append(observation)
    log_probs.append(log_prob)
    actions.append(action)
    rewards.append(reward)
    
    returns = []
    policy_loss = []
    pred_value = []
    target_value = []
    G = 0
    for s, r, log_p, a in zip(reversed(states), reversed(rewards),reversed(log_probs),reversed(actions)):
      returns.append(G)
      G = r + gamma*G
      Vs = vmodel.value(s)
      advantages = (G - float(Vs))      
      policy_loss.append((-1)*(advantages)*log_p)
      pred_value.append(Vs)
      target_value.append(G)
      
      
    policy_loss = torch.cat(policy_loss).sum()
    policy_losses.append(float(policy_loss))
    # update the models
    pmodel.partial_fit(policy_loss)
    vmodel.partial_fit(pred_value,target_value)

plt.plot(totalrewards)
plt.plot(pd.Series(totalrewards).rolling(100).mean())
plt.show()

plt.plot(pd.Series(policy_losses).rolling(100).mean())



pmodel = Policy(D,  16)
vmodel = Value(D,  16)

poptimizer = torch.optim.Adam(pmodel.parameters(), lr=learning_rate)
pmodel.optimizer = poptimizer
voptimizer = torch.optim.Adam(vmodel.parameters(), lr=learning_rate)
vmodel.optimizer = voptimizer


N = 300
for n in range(N):
    reward = 0
    totalreward = 0

    observation = env.reset()
    done = False
    while not done:
        action, log_prob = pmodel.act(observation)#torch.tensor(observation).reshape(1,1,D))        
        prev_observation = observation
        observation, reward, done, info = env.step([action])
        Vs = vmodel.value(prev_observation)
        G = r + gamma*float(Vs)
        advantages = (G - float(Vs))
        
        var = pmodel(torch.tensor(observation))[1]
        
        policy_loss = (-1)*(advantages)*log_prob+0.5*torch.log(2*np.pi*np.e*var)
        pmodel.partial_fit(policy_loss)
        vmodel.partial_fit(vmodel.value(observation),G)
          
        #if reward == 1: # if we changed the reward to -200
        totalreward += reward

plt.plot(totalrewards)
plt.plot(pd.Series(totalrewards).rolling(100).mean())
plt.show()


