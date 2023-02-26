from grid_world import ACTION_SPACE,standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values,print_policy
import numpy as np
from tqdm import tqdm
GAMMA = 0.9
SMALL_ENOUGH = 1e-3


def play_game(grid,policy,max_steps=20):
   start_idx = np.random.choice(len(policy.keys()))
   s = list(policy.keys())[start_idx]
   a = np.random.choice(ACTION_SPACE)
   grid.set_state(s)
   
   states = [s]
   actions = [a]
   rewards = [0]
   
   for _ in range(max_steps):
       r = grid.move(a)
       s = grid.current_state()
       rewards.append(r)
       states.append(s)
       if grid.game_over() :
           break
       a = policy[s]
       actions.append(a)
       
   return states,actions,rewards

if __name__ == '__main__':
     #initialization
    grid = standard_grid()
    print('rewards:')
    print_values(grid.rewards,grid)

    policy = {}
    Q = {}
    returns = {}
    for s in grid.all_states():
        for a in ACTION_SPACE:
            Q[(s,a)] = 0
            returns[(s,a)] = []
        if not grid.is_terminal(s):
            policy[s] = np.random.choice(ACTION_SPACE)

    #loop
    #while True:
    for _ in tqdm(range(10000)):
        states,actions,rewards = play_game(grid,policy)
        states_actions = list(zip(states,actions))
        G = 0
        T = len(states)
        for t in range(T-2,-1,-1):#T-2 para no visitar el terminal state
            r = rewards[t+1]
            s = states[t]
            a = policy[s]
            G = r + GAMMA *G
            #nos quedamos con el valor que toma
            #la primera vez que se visita
            if (s,a) not in states_actions[:t]:
                returns[(s,a)].append(G)
                Q[(s,a)] = np.mean(returns[(s,a)])
                maxQ = float('-inf')
                for a1 in ACTION_SPACE:
                    if maxQ < Q[(s,a1)]:
                        maxQ = Q[(s,a1)]
                        maxa = a1
                policy[s] = maxa
    print('policy:')
    print_policy(policy,grid)
    #find V
    V={}
    for s_a , Qvalue in Q.items():
        s = s_a[0]
        for a1 in ACTION_SPACE:
            if maxQ < Q[(s,a1)]:
                maxQ = Q[(s,a1)]
        V[s] = maxQ
    print('final values')
    print_values(V,grid)