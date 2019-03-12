import numpy as np
import warnings
from random import shuffle
from time import time, sleep
from collections import deque
import os

from settings import s

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.learning_rate = 0.4
    self.discount = 0.7
    self.epsilon = 0.1
    self.train_flag = False
    # TODO: Initialize Jakobs Observation Object
    observation_size = 1
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    try:
        self.q_table = np.load(os.path.join('agent_code', 'merged_agent', 'q_table.npy'))
        self.logger.debug('LOADED Q')
        if self.q_table.shape[1] != 6:
            raise Exception('q_table size does not fit') 
    except Exception as e:
        self.q_table = np.empty([0,6])
        self.logger.info(f'OVERWRITTEN: {e}')

    # Zx10 array with 3x3 observation around agent plus coin_flag
    try:
        self.observation_db = np.load(os.path.join('agent_code', 'merged_agent', 'observation_db.npy'))
        self.logger.debug('LOADED Obs')
    except:
        self.observation_db = np.empty([0,observation_size])
        if self.observation_db.shape[1] != observation_size:
            raise Exception('observation_db size does not fit') 

def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """
    # Default action
    self.next_action = 'WAIT'

    # Gather information about the game state
    # bomb_xys = [(x,y) for (x,y,t) in bombs]
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]

    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    coins = np.array(self.game_state['coins'])
    observation = create_observation(coins, arena, x, y)
    self.old_observation = observation
    self.logger.info(f'self: {[x, y]}')
    self.logger.info(f'Observation: {observation}')    

    # Search for state in observation_db
    # If/else needed because np.where can only be done if self.observation_db is not empty
    if self.observation_db.shape[0] != 0:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
        self.last_q_ind = observation_ind
    else:
        observation_ind = np.array([])

    # If Zufällig und training
        # random action
        # update q_table and observation_db
        # IF observation_db hat keinen Eintrag
            # append q_table and observation_db
    # If Nicht zufällig
        # If observation_db has no entry and training
            # random action
            # q_table and observation_db
        # ElIf observation_db has entry
            # argmax action
        # Else
            # random action

    if self.epsilon > np.random.uniform(0,1) and self.train_flag:
        self.last_action_ind = np.random.randint(0,6)
        if observation_ind.shape[0] == 0:
            self.observation_db = np.append(self.observation_db, np.array([observation]), axis = 0)
            self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
            self.last_q_ind = self.q_table.shape[0] - 1
    else:
        # If state is not yet contained in observation_db it has to be appended and a random action is chosen.
        # Otherwise the action with the highest value is chosen.
        if observation_ind.shape[0] == 0 and self.train_flag:
            self.last_action_ind = np.random.randint(0,6)
            self.observation_db = np.append(self.observation_db, np.array([observation]), axis = 0)
            self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
            self.last_q_ind = self.q_table.shape[0] - 1
        elif observation_ind.shape[0] != 0:
            self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))
        else:
            # Actually regression
            self.last_action_ind = np.random.randint(0,6)

    self.next_action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][self.last_action_ind]

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    self.logger.info(f'Events: {self.events}')

    arena = self.game_state['arena']
    x, y, _, _, _ = self.game_state['self']
    coins = np.array(self.game_state['coins'])
    self.logger.info(f'Coins: {coins.any()}')
    observation = create_observation(coins, arena, x, y)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
    if observation_ind.shape[0] == 0:
        current_best_value = 0
    else:
        current_best_value = self.q_table[observation_ind].max()

    reward = getReward(self.events, self.old_observation)  
    self.q_table[self.last_q_ind, self.last_action_ind] = (1-self.learning_rate) * self.q_table[self.last_q_ind, self.last_action_ind] \
                                                                + self.learning_rate * (reward + self.discount * current_best_value)

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    # Do the same as in reward_update
    reward_update(self)
    np.save(os.path.join('agent_code', 'merged_agent', 'observation_db'), self.observation_db)
    np.save(os.path.join('agent_code', 'merged_agent', 'q_table'), self.q_table)

def getReward(events, old_observation):
    reward = 0
    # Left, right, up, down (0-3) check for coin flag (old_observation[9])
    if 0 in events:
        # if old_observation[9] == 1 or old_observation[3] == 3:
        #     if old_observation[9] == 1: reward += 5 # Reward when agent chooses direction to next coin (coin_flag)
        # else:
            reward -= 5
    elif 1 in events:
        # if old_observation[9] == 2 or old_observation[5] == 3:
        #     if old_observation[9] == 2: reward += 5
        # else:
            reward -= 5
    elif 2 in events:
        # if old_observation[9] == 3 or old_observation[1] == 3:
        #     if old_observation[9] == 3: reward += 5
        # else:
            reward -= 5
    elif 3 in events:
        # if old_observation[9] == 4 or old_observation[7] == 3:
        #     if old_observation[9] == 4: reward += 5
        # else:
            reward -= 5
    # waited
    if 4 in events:
        reward -= 10
    # Interrupted
    if 5 in events:
        reward -= 0
    # Invalid action
    if 6 in events:
        reward -= 500
    # Bomb dropped
    if 7 in events:
        reward -= 500
    # Crate destroyed
    if 9 in events:
        reward += 10
    # Coin found
    if 10 in events:
        reward += 10
    # Coin collected
    if 11 in events:
        reward += 500
    # Killed self
    if 13 in events:
        reward -= 50
    # Got killed
    if 14 in events:
        reward -= 50
    return reward

def determine_flag(best_step, x, y):
    flag = 0
    if best_step == (x-1,y):
        # move left
        flag = 1
    elif best_step == (x+1,y):
        # move right
        flag = 2
    elif best_step == (x,y-1):
        # move up
        flag = 3
    elif best_step == (x,y+1):
        # move down
        flag = 4
    elif best_step == (x,y):
        # Collected???? This case does not make sense. TODO: Find out when it occurs
        flag = 5
    return flag

def create_observation(coins, arena, x, y):
    for c in coins:
        arena[c[0],c[1]] = 3 # Coin

    free_space = arena == 0
    if coins.shape[0] > 0:
        (coin_x, coin_y) = look_for_targets(free_space, (x, y), coins, None)
        coin_flag = determine_flag((coin_x, coin_y), x, y)
    else:
        coin_flag = 0 # no Coin available

    return np.array([coin_flag])

def look_for_targets(free_space, start, targets, logger):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None
    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]