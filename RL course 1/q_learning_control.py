# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:39:19 2022

@author: Fran
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def epsilon_greedy(Q_s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return max_dict(Q_s)[0]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often

  # find max val
  max_val = max(d.values())

  # find keys corresponding to max val
  max_keys = [key for key, val in d.items() if val == max_val]

  ### slow version
  # max_keys = []
  # for key, val in d.items():
  #   if val == max_val:
  #     max_keys.append(key)

  return np.random.choice(max_keys), max_val


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # initialize V(s) and returns
  Q = {}
  states = grid.all_states()
  for s in states:
      Q[s] = {}
      for a in ALL_POSSIBLE_ACTIONS:
          Q[s][a] = 0

  # store max change in V(s) per episode
  rewards_per_episode = []

  # repeat until convergence
  n_episodes = 10000
  for it in range(n_episodes):
    if it % 1000 == 0:
       print("it:", it)
    # begin a new episode
    s = grid.reset()
    total_reward = 0
    while not grid.game_over():
      a = epsilon_greedy(Q[s])
      r = grid.move(a)
      total_reward += r
      s_next = grid.current_state()
      maxQ = max_dict(Q[s_next])[1]
      # update V(s)
      q_old = Q[s][a]
      Q[s][a] = Q[s][a] + ALPHA*(r + GAMMA*maxQ - Q[s][a])  
      #delta = max(delta, np.abs(Q[s][a] - q_old))
      
      # next state becomes current state
      s = s_next

    # store delta
    rewards_per_episode.append(total_reward)

  plt.plot(rewards_per_episode)
  plt.show()


  V = {}
  policy = {}
  states = grid.all_states()
  for s in states:
      V[s] = max_dict(Q[s])[1]
      policy[s] = max_dict(Q[s])[0]


  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)