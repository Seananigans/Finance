import sys
import os
import time
from random import randrange, random, choice

def states(mat):
    return [(i,j) for i in range(len(mat)) for j in range(len(mat[0]))]

def actions(state, environment):
    '''Produce a list of tuples that represent the allowable actions
in the current environment'''
    row_loc = state[0]
    col_loc = state[1]
    acts = []
    if col_loc + 1 < len(environment[0]):
        #RIGHT
        acts.append((0, 1))
    if row_loc + 1 < len(environment):
        #DOWN
        acts.append((1, 0))
    if row_loc - 1 >= 0:
        #UP
        acts.append((-1, 0))
    if col_loc - 1 >= 0:
        #LEFT
        acts.append((0, -1))
    return acts

def reward(state, environment):
    return environment[state[0]][state[1]]

def update(state, actions, environment):
    reward = None
    max_action = actions[randrange(len(actions))] 
    for i in actions:
        current_val = environment[ state[0] + i[0] ][ state[1] + i[1] ]
        if current_val > reward:
            reward = current_val
            max_action = i
    return max_action

def update_q(state, actions, environment, q_table, n_table=None, eps = 0.2):
    max_action = choice(actions)
    reward = q_table[(state, max_action)]
    for i in actions:
        q_value = q_table[(state, i)]
        if q_value > reward:
            reward = q_value
            max_action = i
            
        s_prime_value = environment[ state[0] + i[0] ][ state[1] + i[1] ]
        if s_prime_value > reward:
            reward = s_prime_value
            max_action = i
            
        if (n_table!=None):
            n_value = n_table[(state, i)]
            if n_value > reward:
                reward = n_value
                max_action = i
    if random() < eps:
        max_action = choice(actions)
    return max_action

def table_init(environment, value):
    '''Initialize a q table as dictionary where keys are (state, action)
pairs, and values are initialized to 0.'''
    q = {}
    state = states(environment)
    for s in state:
        acts = actions(s, environment)
        for act in acts:
            q[(s, act)] = q.get((s, act), value)
    return q
    
def play(environment, iterations, sleep=0.5, display=True):
    clear()
    #Initialize discount factor
    gamma = 0.8
    
    #Initialize Learning rate
    alpha = 0.2
    
    #Initialize Q-values table
    q = table_init(environment, 0.)
    #Initialize A table based on how many times a state and action are chosen
    n_table = table_init(environment, 1.)
    
    for i in range(iterations):
        if display==True:
            #Display New Game Message
            new_game_message(0.05)
        
        #Initialize the starting state
        row_loc = randrange(height)
        col_loc = randrange(width)
        state = (row_loc, col_loc)
        
        #Play the game
        while True:
            if display==True:
                draw_agent_location(state, environment, sleep)
            #draw_table(n_table)
            
            #See what actions are available
            acts = actions(state, environment)
            
            #Find action that maximizes reward of next state.
            #act_max = update(state, acts, mat)
            act_max = update_q(state, acts, environment, q, n_table)
            
            n_table[(state, act_max)] /= 2.0
            state_prime = (state[0]+act_max[0], state[1]+act_max[1])
            #Break loop when epoch ends.
            if state_prime == terminal_state:
                q[(state,act_max)] = reward(state_prime, environment)
                break
            
            acts_prime = actions(state_prime, environment)
            #act_max_prime = update(state_prime, acts_prime, mat)
            act_max_prime = update_q(state_prime, acts_prime, environment, q, n_table)
            
            #Update Q table value for State, Action pair
            q[(state, act_max)] += alpha*(
                reward(state_prime, environment) + \
                gamma*(q[(state_prime, act_max_prime)]) - \
                q[(state, act_max)]
                )
            
            #Update state to be new state
            previous_action = act_max
            state = state_prime
    time.sleep(sleep)
    clear()
    draw_table(q)
    

def draw_table(q):
    '''Draws the Q table'''
    print "Q table:"
    print "((state),(action))\tvalue"
    q_sort = sorted(q)
    for k in q_sort:
        if k[0] != terminal_state:
            print k, "\t", q[k]

def draw_agent_location(state, environment, sleep=0.5):
    time.sleep(sleep)
    clear()
    for row in range(len(environment)):
        line = ""
        for col in range(len(environment[0])):
            if (row, col) == state:
                line += "*" + "\t"
            elif (row, col) == terminal_state:
                line += "!!" + "\t"
            else:
                line += "_\t"#str(environment[row][col]) + "\t"
        print line

def new_game_message(message_duration = 0.5):
    print """
NNN   NN  EEEEEEE  WW     WW
NNNN  NN  EE       WW     WW
NN NN NN  EEEE     WW WWW WW
NN  NNNN  EE       WWW   WWW
NN   NNN  EEEEEEE  WW     WW
"""
    time.sleep(message_duration)

def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

class Agent():
    def __init__(self, environment):
        self.environment = environment
        height = len(environment)
        width = len(environment[0])
        self.state = (randrange(height), randrange(width))
        #self.acts = self.actions(self.state)
        self.q_table = {}
        self.gamma= 0.5
        self.alpha = 1.0
    
    def policy(self, state, epsilon=0.1):
        #Get actions for state
        acts = self.actions(state)
        reward = None
        for i in acts:
            self.q_table[(state, i)] = self.q_table.get((state, i), 0.0)
            if self.q_table[(state, i)] > reward:
                reward = self.q_table[(state, i)]
                max_action = i
        if random()<epsilon:
            max_action = choice(acts)
        return max_action

    def execute_action(self, state, action):
        state_prime = (state[0] + action[0], state[1] + action[1])
        return state_prime

    def update(self):
        action = self.policy(self.state)
        state_prime = self.execute_action(self.state, action)
        action_prime = self.policy(state_prime)
        #Update Q table with state, action, reward, state_prime, and action_prime
        q_prime = self.q_table[(state_prime, action_prime)]
        value = self.reward(self.state) + self.gamma*q_prime
        value -= self.q_table[(self.state, action)]
        self.q_table[(self.state, action)] += self.alpha* value
        #Update state of agent to be new state_prime
        self.state = state_prime
        
    def hallucinate(self, iterations):
        for i in range(iterations):
            #Choose random state from q_table and act randomly
            s, a = choice(self.q_table.keys())
            #a = choice(self.actions(s))
            s_prime = self.execute_action(s, a)
            a_prime = self.policy(s_prime)
            #Update Q table with state, action, reward, state_prime, and action_prime
            q_prime = self.q_table[(s_prime, a_prime)]
            value = self.reward(s) + self.gamma*q_prime - self.q_table[(s, a)]
            self.q_table[(s, a)] += self.alpha * value
    
    def actions(self, state):
        '''Produce a list of tuples that represent the allowable actions
    in the current environment'''
        row_loc = state[0]
        col_loc = state[1]
        acts = []
        if col_loc + 1 < len(self.environment[0]):
            #RIGHT
            acts.append((0, 1))
        if row_loc + 1 < len(self.environment):
            #DOWN
            acts.append((1, 0))
        if row_loc - 1 >= 0:
            #UP
            acts.append((-1, 0))
        if col_loc - 1 >= 0:
            #LEFT
            acts.append((0, -1))
        return acts

    def reward(self, state):
        return self.environment[state[0]][state[1]]
        

if __name__ == "__main__":
    height = 5
    width = 5

    init_rew = 0.0
    mat = [[init_rew for i in range(width)] for j in range(height)]
    row_loc = randrange(height)
    col_loc = randrange(width)
    terminal_state = (row_loc, col_loc)
    #terminal_state = (1, 2)

    mat[terminal_state[0]][terminal_state[1]] = 100.

    display = True
    number_of_games = 500000
    number_of_games = int(raw_input("How many games would you like the robot to play? "))
    if number_of_games > 100:
        display = False
        
    play(mat, number_of_games, 0.5, display)

    agent = Agent(mat)
    for i in range(100):
        agent.update()
        agent.hallucinate(100)
    draw_table(agent.q_table)
