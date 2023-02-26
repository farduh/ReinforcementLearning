from grid_world import windy_grid,ACTION_SPACE
import numpy as np

SMALL_ENOUGH = 1e-5


def print_values(V,g):
    for i in range(g.rows):
        print('--------------------')
        for j in range(g.cols):
            v = V.get((i,j),0)
            if v >= 0:
                print(f' {v:.3f}|',end='')
            else:
                print(f'{v:.3f}|',end='')#signo negativo ocupa un lugar extra
        print('')

def print_policy(P,g):
    for i in range(g.rows):
        print('----------------')
        for j in range(g.cols):
            a = P.get((i,j),' ')
            print(f' {a} |',end='')
        print('')
 



transitions_probs = {}
rewards = {}
grid = windy_grid()

for (s,a),v in grid.probs.items():
    for s2, p in v.items():
        transitions_probs[(s,a,s2)] = p
        rewards[(s,a,s2)] = grid.rewards.get(s2,0)
    
policy = {
    (2,0):{'U':0.5,'R':0.5},
    (1,0):{'U':1},
    (0,0):{'R':1},
    (0,1):{'R':1},
    (0,2):{'R':1},
    (1,2):{'U':1},
    (2,1):{'R':1},
    (2,2):{'U':1},
    (2,3):{'L':1}
    }

print_policy(policy,grid)

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
                    action_prob = policy[s].get(a,0)
                    
                    r = rewards.get((s,a,s2),0)
                    new_v += action_prob * transitions_probs.get((s,a,s2),0)*(r + gamma * V[s2])
                    
            V[s] = new_v
            biggest_change = max(biggest_change,np.abs(old_v-V[s]))

    print('iter',it,'biggest_change',biggest_change)
    print_values(V,grid)
    it += 1
    if biggest_change < SMALL_ENOUGH:
        break