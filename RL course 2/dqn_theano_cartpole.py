#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:18:22 2022

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
        W_init = np.random.randn(M1,M2)/np.sqrt(M2+M1)
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
    
class DQN:
    def __init__(self, D, K, hidden_layer_sizes, gamma, 
                 max_experiences=10000,min_experiences=100, batch_sz=100):
        self.K =K
        lr = 1e-2
        mu = 0.1
        decay = 0.99
        
        #create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2
        
        #final layer
        layer = HiddenLayer(M1, K, lambda x:x)
        self.layers.append(layer)
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
        caches = [th.shared(np.ones_like(p.get_value())*0.1) for p in self.params]
        velocities = [th.shared(p.get_value()*0.) for p in self.params]
    
        #inputs and targets
        X = T.matrix('X')
        G = T.vector('G')
        actions = T.ivector('actions')
    
        #calculate output and cost
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        
        selected_action_values = Y_hat[T.arange(actions.shape[0]),actions]
        cost = T.sum((G-selected_action_values)**2)
        
        #create train function
        grads = T.grad(cost,self.params)
        g_update = [(p,p+v) for p, v, g in zip(self.params,velocities,grads)]
        c_update = [(c,decay*c+(1-decay)*g*g) for c, g in zip(caches,grads)]
        v_update = [(v,mu*v-lr*g/T.sqrt(c)) for v, c, g in zip(velocities, caches,grads)]
    
        updates = g_update + c_update + v_update
        
        #compile function
        self.train_op = th.function(
            inputs=[X,G,actions],
            updates=updates,
            allow_input_downcast=True)
        
        self.predict_op = th.function(
            inputs=[X],
            outputs=Y_hat,
            allow_input_downcast=True)
        
        self.experience = {'s':[],'a':[],'r':[],'s2':[]}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma
    
    def copy_from(self,other):
         my_params = self.params
         other_params = other.params
         for p,q in zip(my_params,other_params):
             actual = q.get_value()
             p.set_value(actual)
             
    def predict(self,X):
        X = np.atleast_2d(X)
        return self.predict_op(X)
    
    def train(self,target_network):
        
        if len(self.experience['s']) < self.min_experiences:
            return
        
        idx = np.random.choice(len(self.experience['s']),size=self.batch_sz,
                               replace=False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states),axis=1)
        targets = [r + self.gamma*next_q for r,next_q in zip(rewards, next_Q)]
        
        #call optimizer
        self.train_op(states,targets,actions)
        
    def add_experience(self,s,a,r,s2):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])
        
def play_one(env, model, tmodel, eps, gamma, copy_period):
    observation = env.reset()
    done = False
    totalreward = 0 
    iters = 0
    while not done and iters< 2000:
        action = model.sample_action(observation,eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        
        totalreward += reward
        if done:
            reward = -200
        
        #update model
        model.add_experience(prev_observation,action,reward,observation)
        model.train(tmodel)
        
        iters+=1
        
        if iters % copy_period == 0:
            tmodel.copy_from(model)
        
    return totalreward

def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50
    
    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [100,50]
    model = DQN(D,K,sizes,gamma)
    tmodel = DQN(D,K,sizes,gamma)
    
    if 'monitor' in sys.argv:
      filename = os.path.basename(__file__).split('.')[0]
      monitor_dir = './' + filename + '_' + str(datetime.now())
      env = wrappers.Monitor(env, monitor_dir)
    
    N = 6000
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1./np.sqrt(n+1)
        totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
       
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())
       
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
       
    plot_running_avg(totalrewards)

if __name__ == '__main__':
    main()
    
   