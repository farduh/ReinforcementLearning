from grid_world import ACTION_SPACE,standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values,print_policy
import numpy as np

GAMMA = 0.9
SMALL_ENOUGH = 1e-3


    
def play_game(grid,policy,max_steps=20):
   start_idx = np.random.choice(len(policy.keys()))
   s = list(policy.keys())[start_idx]
   grid.set_state(s)
   
   states = [s]
   rewards = [0]
   
   steps = 0
   while not grid.game_over():
       a = policy[s]
       r = grid.move(a)
       rewards.append(r)
       s = grid.current_state()
       states.append(s)    
       steps += 1
       if steps >= max_steps:
           break
   return states,rewards

if __name__ == '__main__':

    grid = standard_grid()
        
    transitions_probs = {}
    rewards = {}
    
    # policy = {
    #     (2,0):'U',
    #     (1,0):'U',
    #     (0,0):'R',
    #     (0,1):'R',
    #     (0,2):'R',
    #     (1,2):'U',
    #     (2,1):'R',
    #     (2,2):'U',
    #     (2,3):'L'
    #     }
    
    policy = {
        (2,0):'U',
        (1,0):'U',
        (0,0):'R',
        (0,1):'R',
        (0,2):'R',
        (1,2):'R',
        (2,1):'R',
        (2,2):'R',
        (2,3):'U'
        }
    
    V = {}
    returns = {}
    for s in grid.all_states():
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0    
    for _ in range(100):
       #play episode
       states,rewards = play_game(grid,policy)
       #value evaluation
       G = 0
       T = len(states)
       for t in range(T-2,-1,-1):#T-2 para no visitar el terminal state
           r = rewards[t+1]
           s = states[t]
           G = r + GAMMA *G
           #nos quedamos con el valor que toma
           #la primera vez que se visita
           if s not in s[:t]:
               returns[s].append(G)
               V[s] = np.mean(returns[s])
       
        
    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)
    
           
   
    
    

