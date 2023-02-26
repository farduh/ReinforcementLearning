#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:19:59 2022

@author: farduh
"""

import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


    

class PolicyModel(nn.Module):
    
    def __init__(self,n_inputs, n_outputs, hidden_layers,loss_fn,optimizer):
        super(PolicyModel, self).__init__()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        neurons_by_layers = [n_inputs]+hidden_layers+[n_outputs]
        self.layers = nn.ModuleList()
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        for i in range(1,len(neurons_by_layers)):
            self.layers.append(torch.nn.Linear(neurons_by_layers[i-1], neurons_by_layers[i]))
        
    def forward(self,x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.softmax(x)
        return x

    def act(self, state):
        # p = self(X)[0][0]
        # nonans = torch.all(~torch.isnan(p))
        # #assert(nonans)
        # if not nonans:
        #     return 0,torch.log(p)
        # idx = p.multinomial(num_samples=1, replacement=True)
        # return int(idx[0]),torch.log(p)
    
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    def partial_fit(self,loss):
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    def partial_fit(self,loss):
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
class ValueModel(nn.Module):
    
    def __init__(self,n_inputs, n_outputs, hidden_layers,loss_fn,optimizer):
        super(ValueModel, self).__init__()
        
        neurons_by_layers = [n_inputs]+hidden_layers+[n_outputs]
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.layers = nn.ModuleList()
        self.activation = torch.nn.ReLU()
        for i in range(1,len(neurons_by_layers)):
            self.layers.append(torch.nn.Linear(neurons_by_layers[i-1], neurons_by_layers[i]))
        
    def forward(self,x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    
    def partial_fit(self,state,G):
        pred = self(state)
        loss = self.loss_fn(pred, G)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



learning_rate = 1e-2
gamma = 1

env = gym.make('CartPole-v0')
D = env.observation_space.shape[0]
K = env.action_space.n
pmodel = PolicyModel(D, K, [16,2],None,None)

pmodel = Policy(D, K, 16)

vmodel = ValueModel(D,1, [16],None,None)
optimizer = torch.optim.Adam(pmodel.parameters(), lr=learning_rate)
pmodel.optimizer = optimizer
vmodel.optimizer = torch.optim.Adam(vmodel.parameters(), lr=learning_rate)
vmodel.loss_fn = nn.MSELoss()

N = 3000
totalrewards = []

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
        observation, reward, done, info = env.step(action)
        if done:
          reward = -500
    
        if reward == 1: # if we changed the reward to -200
          totalreward += reward
    totalrewards.append(totalreward)
    action, log_prob = pmodel.act(observation)#torch.tensor(observation).reshape(1,1,D))
    states.append(observation)
    log_probs.append(log_prob)
    actions.append(action)
    rewards.append(reward)
    
    returns = []
    policy_loss = []
    G = 0
    for s, r, log_p, a in zip(reversed(states), reversed(rewards),reversed(log_probs),reversed(actions)):
      returns.append(G)
      G = r + gamma*G
      advantages = (G - float(vmodel(torch.tensor(s))[0]))      
      
      policy_loss.append((-1)*advantages*log_p)
    returns.reverse()
    #policy_targets.reverse()
    policy_loss = torch.cat(policy_loss).sum() 
    # update the models
    vmodel.partial_fit(torch.tensor(states[:-1]).reshape(-1,1,D), torch.tensor(returns[1:]).reshape(-1,1,1))
    pmodel.partial_fit(policy_loss)

plt.plot(totalrewards)




env = gym.make('CartPole-v0')
D = env.observation_space.shape[0]
K = env.action_space.n

pmodel = Policy(D, K, 16)
vmodel = ValueModel(D,1, [16],None,None)
optimizer = torch.optim.Adam(pmodel.parameters(), lr=learning_rate)
pmodel.optimizer = optimizer
vmodel.optimizer = torch.optim.Adam(vmodel.parameters(), lr=learning_rate)
vmodel.loss_fn = nn.MSELoss()
learning_rate = 1e-3
gamma = 1

N = 3000
totalrewards = []
mean_advantages = []
for n in range(N):
    reward = 0
    totalreward = 0
    observation = env.reset()
    done = False
    advantages = []
    while not done:
        action, log_prob = pmodel.act(observation)#torch.tensor(observation).reshape(1,1,D))
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        if done:
          reward = -200
    
        if reward == 1: # if we changed the reward to -200
          totalreward += reward
        G = 0
        Vst1 = float(vmodel(torch.tensor(observation))[0])
        Vst0 = float(vmodel(torch.tensor(prev_observation))[0])
        G = reward + gamma*Vst1
        
        advantage = (G - Vst0)
        policy_loss =  (-1)*advantage*log_prob
        #policy_loss = torch.cat(policy_loss).sum()
        if not n%2:
            vmodel.partial_fit(torch.tensor(prev_observation).reshape(-1,1,D),torch.tensor(G).reshape(-1,1,1) )
        else:
            pmodel.partial_fit(policy_loss)
        advantages.append(advantage)
    mean_advantages.append(np.mean(advantages))
    totalrewards.append(totalreward)

plt.plot(totalrewards)

plt.plot((np.array(mean_advantages)**2))