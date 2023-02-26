import numpy as np
ACTION_SPACE = ['U','D','L','R']

class Grid():
    def __init__(self,rows,cols,start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
    
    def set(self,rewards,actions):
        self.rewards = rewards
        self.actions = actions
    
    def set_state(self,s):
        self.i = s[0]
        self.j = s[1]
    
    def current_state(self):
        return (self.i,self.j)
    
    def is_terminal(self,s):
        return s not in self.actions
    
    def reset(self):
      # put agent back in start position
      self.i = 2
      self.j = 0
      return (self.i, self.j)
    
    def get_next_state(self,s,a):
        i, j = s[0], s[1]
        if a in self.actions[(i,j)]:
            if a == 'U':
                i-=1
            elif a == 'D':
                i+=1
            elif a == 'R':
                j+=1
            elif a == 'L':
                j-=1
        return i,j
    
    def move(self,action):
        if action in self.actions[(self.i,self.j)]:
            if action == 'U':
                self.i-=1
            elif action == 'D':
                self.i+=1
            elif action == 'R':
                self.j+=1
            elif action == 'L':
                self.j-=1
        return self.rewards.get((self.i,self.j),0)
    
    def undo_move(self,action):
        if action == 'U':
            self.i+=1
        elif action == 'D':
            self.i-=1
        elif action == 'R':
            self.j-=1
        elif action == 'L':
            self.j+=1
        assert (self.current_state() in self.all_states())
    
    def game_over(self):
        return (self.i,self.j) not in self.actions
    
    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

def standard_grid():
    g = Grid(3,4,(2,0))
    rewards = {(0,3):1,(1,3):-1}
    actions = {
        (0,0):('D','R'),
        (0,1):('L','R'),
        (0,2):('L','D','R'),
        (1,0):('U','D'),
        (1,2):('U','D','R'),
        (2,0):('U','R'),
        (2,1):('L','R'),
        (2,2):('L','R','U'),
        (2,3):('L','U')}
    g.set(rewards,actions)
    return g


class WindyGrid(Grid):
    
    def set(self,rewards,actions,probs):
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def move(self,action):
        s = (self.i,self.j)
        a = action
        
        next_state_probs = self.probs[(s,a)]
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        s2 = np.random.choice(next_states, p=next_probs)
        
        self.i, self.j = s2
        
        return self.rewards.get(s2,0)
    
def windy_grid():
    g = WindyGrid(3,4,(2,0))
    rewards = {(0,3):1,(1,3):-1}
    actions = {
        (0,0):('D','R'),
        (0,1):('L','R'),
        (0,2):('L','D','R'),
        (1,0):('U','D'),
        (1,2):('U','D','R'),
        (2,0):('U','R'),
        (2,1):('L','R'),
        (2,2):('L','R','U'),
        (2,3):('L','U')}
    
    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
      }
    g.set(rewards, actions, probs)
    return g

def windy_grid_no_wind():
  g = windy_grid()
  g.probs[((1, 2), 'U')] = {(0, 2): 1.0}
  return g



def windy_grid_penalized(step_cost=-0.1):
  g = WindyGrid(3, 4, (2, 0))
  rewards = {
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0, 3): 1,
    (1, 3): -1
  }
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }

  # p(s' | s, a) represented as:
  # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
  probs = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  g.set(rewards, actions, probs)
  return g



def grid_5x5(step_cost=-0.1):
  g = Grid(5, 5, (4, 0))
  rewards = {(0, 4): 1, (1, 4): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'R'),
    (0, 3): ('L', 'D', 'R'),
    (1, 0): ('U', 'D', 'R'),
    (1, 1): ('U', 'D', 'L'),
    (1, 3): ('U', 'D', 'R'),
    (2, 0): ('U', 'D', 'R'),
    (2, 1): ('U', 'L', 'R'),
    (2, 2): ('L', 'R', 'D'),
    (2, 3): ('L', 'R', 'U'),
    (2, 4): ('L', 'U', 'D'),
    (3, 0): ('U', 'D'),
    (3, 2): ('U', 'D'),
    (3, 4): ('U', 'D'),
    (4, 0): ('U', 'R'),
    (4, 1): ('L', 'R'),
    (4, 2): ('L', 'R', 'U'),
    (4, 3): ('L', 'R'),
    (4, 4): ('L', 'U'),
  }
  g.set(rewards, actions)

  # non-terminal states
  visitable_states = actions.keys()
  for s in visitable_states:
    g.rewards[s] = step_cost

  return g


def negative_grid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  return g
