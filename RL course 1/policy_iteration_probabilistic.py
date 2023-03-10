# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:32:32 2021

@author: Fran
"""

from grid_world import ACTION_SPACE,windy_grid_penalized,windy_grid
from iterative_policy_evaluation_deterministic import print_values,print_policy
import numpy as np

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

def get_transition_probs_and_rewards(grid):
        
    transitions_probs = {}
    rewards = {}
    
    for (s,a),v in grid.probs.items():
        for s2, p in v.items():
            transitions_probs[(s,a,s2)] = p
            rewards[(s,a,s2)] = grid.rewards.get(s2,0)
        
    return transitions_probs,rewards


def evaluate_deterministic_policy(grid,policy,transitions_probs,rewards,V=None):
    if not V:
        V = {}
        for s in grid.all_states():
            V[s] = 0
            
    gamma = 0.9
    
    it = 0
    while True:
        biggest_change = 0 
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        
                        #action is deterministic
                        action_prob = 1 if policy.get(s) == a else 0
                        
                        r = rewards.get((s,a,s2),0)
                        new_v += action_prob * transitions_probs.get((s,a,s2),0)*(r + gamma * V[s2])
                        
                V[s] = new_v
                biggest_change = max(biggest_change,np.abs(old_v-V[s]))
    
        it += 1
        if biggest_change < SMALL_ENOUGH:
            break
    return V

def init_policy(grid):
    policy = {}
    for s in grid.all_states():
        policy[s] = np.random.choice(ACTION_SPACE)
    return policy

if __name__ == '__main__':
    
    grid = windy_grid_penalized(-0.1)
    transitions_probs,rewards = get_transition_probs_and_rewards(grid)
    print('rewards:')
    print_values(grid.rewards,grid)
    policy = init_policy(grid)
    print('inicial policy:')
    print_policy(policy,grid)
    V = None
    while True:
        V = evaluate_deterministic_policy(grid,policy,transitions_probs,rewards,V)
        is_policy_converged = True
        for s in grid.actions.keys():
            old_a = policy[s]
            new_a = None
            best_value = float("-inf")
            
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    r = rewards.get((s,a,s2),0)
                    v += transitions_probs.get((s,a,s2),0)* (r + GAMMA * V[s2])
                    
                if v > best_value:
                    best_value = v
                    new_a = a
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False
        if is_policy_converged:
            break
        
    print('values:')
    print_values(V,grid)
    print('policy:')
    print_policy(policy,grid)
        
