# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:20:30 2022

@author: Fran
"""

import gym
from gym import wrappers
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '.\\video\\')
steps_in_episodes = []
for _ in range(10000):
    state = env.reset()
    done = False
    i = 0
    while not done:
        random_action = env.action_space.sample()
        state, reward, done, info = env.step(random_action)
        i+=1
        
    steps_in_episodes.append(i)
    
plt.hist(steps_in_episodes)
plt.yscale('log')