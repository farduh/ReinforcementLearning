#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:50:02 2022

@author: farduh
"""

import gym
import os
import sys
import numpy as np
import theano as th
import theano.tensor as T
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning import plot_running_avg



class HiddenLayer():
    def __init__(self,M1,M2,f=T.tanh,use_bias=True):
        W_init = np.random.randn(M1,M2)*np.sqrt(2/M1)
        self.W = th.shared(W_init)
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            b_init = np.zeros(M2)
            self.b = th.shared(b_init)
            self.params += [self.b]
        self.f = f

    def forward(self,X):
        if self.use_bias:
            a = X.dot(self.W) + self.b
        else:
            a = X.dot(self.W)
        return self.f(a)

class PolicyModel():
    def __init__(self,D,K,hidden_layer_sizes):
        lr= 1e-4
        mu = 0.7
        decay = 0.999
        
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        
        #final layer
        layer = HiddenLayer(M1, K,lambda x: x, use_bias=False)
        self.layers.append(layer)
        
        params = []      
        for layer in self.layers:
            params += layer.params
        
        caches = [th.shared(np.ones_like(p.get_value())*0.1) for p in params]
        velocities = [th.shared(p.get_value()*0.) for p in params]
        
        #input and targets
        X = T.matrix('X')
        actions = T.ivector('actions')
        advantages = T.vector('advantages')
        
        Z=X
        for layer in self.layers:
            Z = layer.forward(Z)
        action_score = Z
        p_a_given_s = T.nnet.softmax(action_score)
        
        selected_probs = T.log(p_a_given_s[T.arange(actions.shape[0]),actions])
        cost = -T.sum(advantages*selected_probs)
        
        grads = T.grad(cost,params)
        updates = [(p, p - lr*g) for p, g in zip(params, grads)]

        # g_update = [(p,p+v) for p,v,g in zip(params,velocities,grads)]
        # c_update = [(c,decay*c + (1-decay)*g*g) for c,g in zip(caches,grads)]
        # v_update = [(v,mu*v - lr*g/T.sqrt(c)) for v,c,g in zip(velocities,caches,grads)]
        # #v_update = [(v,mu*v - lr*g) for v,c,g in zip(velocities,grads)]
        
        # updates = c_update + g_update + v_update
        
        #compile function
        self.train_op = th.function(
            inputs=[X,actions,advantages],
            updates=updates,
            allow_input_downcast=True)
        
        self.predict_op = th.function(
            inputs=[X],
            outputs=p_a_given_s,
            allow_input_downcast=True)
    
    def partial_fit(self,X,actions,advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.train_op(X,actions,advantages)
    
    def predict(self,X):
        X = np.atleast_2d(X)
        return self.predict_op(X)
    
    def sample_action(self,X):
        p = self.predict(X)[0]
        nonans = np.all(~np.isnan(p))
        assert (nonans)
        return np.random.choice(len(p),p=p)

class ValueModel:
    def __init__(self,D,hidden_layer_sizes):
        lr= 1e-4
        
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        
        #final layer
        layer = HiddenLayer(M1, 1,lambda x: x)
        self.layers.append(layer)
        
        params = []    
        for layer in self.layers:
            params += layer.params
        
        #input and targets
        X = T.matrix('X')
        Y = T.vector('Y')
        
        Z=X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = T.flatten(Z)
        cost = T.sum((Y-Y_hat)**2)
        
        grads = T.grad(cost,params)
        updates = [(p,p-lr*g) for p,g in zip(params,grads)]
        
        #compile function
        self.train_op = th.function(
            inputs=[X,Y],
            updates=updates,
            allow_input_downcast=True)
        
        self.predict_op = th.function(
            inputs=[X],
            outputs=Y_hat,
            allow_input_downcast=True)
        
    def partial_fit(self,X,Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.train_op(X,Y)
    
    def predict(self,X):
        X = np.atleast_2d(X)
        return self.predict_op(X)
    

def play_one_td(env,pmodel,vmodel,gamma):
    observation =  env.reset()
    done = False
    totalreward = 0
    iters = 0
    
    while not done and iters< 2000:
        
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        
        V_next = vmodel.predict(observation)[0]
        G = reward+gamma*V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation,action,advantage)
        vmodel.partial_fit(observation,G)
 
        # if done:
        #     reward = -200
        
        
        if reward == 1:
            totalreward += reward
        iters+=1
    return totalreward

    
def play_one_mc(env,pmodel,vmodel,gamma):
    observation =  env.reset()
    done = False
    totalreward = 0
    iters = 0
    
    states = []
    actions = []
    rewards = []
    
    while not done and iters< 2000:
        
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        
        if done:
            reward = -200
        
        states.append(prev_observation)
        actions.append(action)
        rewards.append(reward)
        
        if reward == 1:
            totalreward += reward
        iters+=1
    
    returns = []
    advantages = []
    G = 0
    
    for s, r in zip(reversed(states),reversed(rewards)):
        returns.append(G)
        advantages.append(G-vmodel.predict(s)[0])
        G = r+ gamma*G
    
    returns.reverse()
    advantages.reverse()
    
    pmodel.partial_fit(states,actions,advantages)
    vmodel.partial_fit(states,returns)
    
    return totalreward

def main():
    env = gym.make('CartPole-v0') 
    D = env.observation_space.shape[0]
    K = env.action_space.n
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    gamma = 0.99
    
    if 'monitor' in sys.argv:
      filename = os.path.basename(__file__).split('.')[0]
      monitor_dir = './' + filename + '_' + str(datetime.now())
      env = wrappers.Monitor(env, monitor_dir)

    N = 200000
    totalrewards = np.empty(N)
    for n in range(N):
        totalreward = play_one_td(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
    print('avg reward for last 100 episodes:',totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

if __name__ == '__main__':
  main()