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
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    try:
        self.q_table = np.load(os.path.join('agent_code', 'eike_agent', 'q_table_cccb.npy'))
        self.logger.debug('LOADED Q')
    except Exception as e:
        self.q_table = np.empty([0,6])
        self.logger.info(f'OVERWRITTEN: {e}')

    # Zx10 array with 3x3 observation around agent plus coin_flag
    try:
        self.observation_db = np.load(os.path.join('agent_code', 'eike_agent', 'observation_db_cccb.npy'))
        self.logger.debug('LOADED Obs')
    except:
        self.observation_db = np.empty([0,13])
    self.learning_rate = 0.4
    self.discount = 0.7
    self.epsilon = 0.2
    self.train_flag = True

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
    self.next_action = 'RIGHT'

    # Gather information about the game state
    # bomb_xys = [(x,y) for (x,y,t) in bombs]
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]

    
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    coins = np.array(self.game_state['coin_locs'])
    bombs = self.game_state['bombs']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    danger_zone_ind = np.where(bomb_map < 5)
    danger_zone_coords = np.vstack((danger_zone_ind[0], danger_zone_ind[1])).T
    is_on_danger_zone = (danger_zone_coords == np.array([x,y])).all(axis=1).any()
    # explosions = self.game_state['explosions']
    # explosions_ind1 = np.where(explosions == 2)
    # explosions_ind2 = np.where(explosions == 1)
    # explosions_coords = np.vstack((np.concatenate((explosions_ind1[0], explosions_ind2[0])), np.concatenate((explosions_ind1[1], explosions_ind2[1])))).T
    observation = create_observation(coins, arena, danger_zone_coords, is_on_danger_zone, x, y)
    self.old_observation = observation
    self.logger.info(f'self: {[x, y]}')
    self.logger.info(f'Observation: {observation}')
    # self.logger.info(f'Explosions: {explosions}')
    # self.logger.info(f'Bomb map: {bomb_map}')
    self.logger.info(f'Danger zone coords: {danger_zone_coords}')
    self.logger.info(f'Is on danger zone: {is_on_danger_zone}')
    

    # Search for state in observation_db
    # If/else needed because np.where can only be done if self.observation_db is not empty
    if self.observation_db.shape[0] != 0:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
        self.last_q_ind = observation_ind
    else:
        observation_ind = np.array([])

    if self.epsilon > np.random.uniform(0,1) and self.train_flag:
        self.last_action_ind = np.random.randint(0,6)
    else:
        # If state is not yet contained in observation_db it has to be appended and a random action is chosen.
        # Otherwise the action with the highest value is chosen.
        if observation_ind.shape[0] == 0 and self.train_flag:
            self.observation_db = np.append(self.observation_db, np.array([observation]), axis = 0)
            self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
            self.last_action_ind = np.random.randint(0,6)
            self.last_q_ind = self.q_table.shape[0] - 1
        else:
            if observation_ind.shape[0] == 0:
                self.last_action_ind = np.random.randint(0,6)
            else:
                self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))
                self.logger.info(f'Q-TABLE: {self.q_table[observation_ind[0]] }')

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
    self.logger.info(f'self: {[x, y]}')
    coins = np.array(self.game_state['coin_locs'])
    self.logger.info(f'Coins: {coin_locs.any()}')
    bombs = self.game_state['bombs']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    danger_zone_ind = np.where(bomb_map < 5)
    danger_zone_coords = np.vstack((danger_zone_ind[0], danger_zone_ind[1])).T
    is_on_danger_zone = (danger_zone_coords == np.array([x,y])).all(axis=1).any()
    # explosions = self.game_state['explosions']
    # explosions_ind1 = np.where(explosions == 2)
    # explosions_ind2 = np.where(explosions == 1)
    # explosions_coords = np.vstack((np.concatenate((explosions_ind1[0], explosions_ind2[0])), np.concatenate((explosions_ind1[1], explosions_ind2[1])))).T
    observation = create_observation(coins, arena, danger_zone_coords, is_on_danger_zone, x, y)
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
    np.save(os.path.join('agent_code', 'eike_agent', 'observation_db_cccb'), self.observation_db)
    np.save(os.path.join('agent_code', 'eike_agent', 'q_table_cccb'), self.q_table)

def getReward(events, old_observation):
    reward = 0
    # Left, right, up, down (0-3) check for coin flag (old_observation[9])
    if 0 in events:
        if old_observation[9] == 1 or old_observation[3] == 3:
            if old_observation[9] == 1: reward += 5 # Reward when agent chooses direction to next coin (coin_flag)
            if old_observation[3] == 3: reward += 10 # Reward when agent chooses direction to visible coin in 1 field radius
            if old_observation[10] == 1: reward += 2 # Reward when agent chooses direction to next crate (crate_flag)
        else:
            reward -= 2
    elif 1 in events:
        if old_observation[9] == 2 or old_observation[5] == 3:
            if old_observation[9] == 2: reward += 5
            if old_observation[5] == 3: reward += 10
            if old_observation[10] == 2: reward += 2 

        else:
            reward -= 2
    elif 2 in events:
        if old_observation[9] == 3 or old_observation[1] == 3:
            if old_observation[9] == 3: reward += 5
            if old_observation[1] == 3: reward += 10
            if old_observation[10] == 3: reward += 2
        else:
            reward -= 2
    elif 3 in events:
        if old_observation[9] == 4 or old_observation[7] == 3:
            if old_observation[9] == 4: reward += 5
            if old_observation[7] == 3: reward += 10
            if old_observation[10] == 4: reward += 2
        else:
            reward -= 2
    # waited
    elif 4 in events:
        reward -= 5
    # Interrupted
    if 5 in events:
        reward -= 0
    # Invalid action
    if 6 in events:
        reward -= 50
    # Bomb dropped
    if 7 in events:
        if old_observation[1] == 1 or old_observation[3] == 1 or old_observation[5] == 1 or old_observation[7] == 1:
            reward += 20
        else:
            reward -= 50
    # Crate destroyed
    if 9 in events:
        reward += 10
    # Coin found
    if 10 in events:
        reward += 10
    # Coin collected
    if 11 in events:
        reward += 50
    # Killed self
    if 13 in events:
        reward -= 50
    # Killed self
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

def create_observation(coins, arena, danger_zone_coords, is_on_danger_zone, x, y):
    for c in coins:
        arena[c[0],c[1]] = 3 # Coin
    
    # Ignore danger zones which are on walls TODO: also those inside the field
    danger_zone_coords = danger_zone_coords[((danger_zone_coords > 0) & (danger_zone_coords < 16)).all(axis=1)]
    arena[danger_zone_coords[:, 0], danger_zone_coords[:, 1]] = 10 # Danger zone

    free_space = arena == 0
    # logger.info(f'Free Space: {free_space, type(free_space)}')
    leave_danger_zone_flag = 0
    if is_on_danger_zone:
        (leave_danger_zone_x, leave_danger_zone_y) = look_for_targets(free_space, (x, y), free_space, None)
        leave_danger_zone_flag = determine_flag((leave_danger_zone_x, leave_danger_zone_y), x, y)

    if(bool(coins.any())):
        (coin_x, coin_y) = look_for_targets(free_space, (x, y), coins, None)
        coin_flag = determine_flag((coin_x, coin_y), x, y)
    else:
        coin_flag = 0 # no Coin available
    
    crates_ind = np.where(arena == 1)
    crate_coords = np.vstack((crates_ind[0], crates_ind[1])).T
    if(bool(crate_coords.any())):
        (crate_x, crate_y) = look_for_targets(free_space, (x,y), crate_coords, None)
        crate_flag = determine_flag((crate_x, crate_y), x, y)
    else:
        crate_flag = 0 # no Crate available

    return np.array([arena[x-1, y-1], arena[x, y-1], arena[x+1, y-1], arena[x-1, y], arena[x, y], arena[x+1, y], arena[x-1, y+1], arena[x, y+1], arena[x+1, y+1], coin_flag, crate_flag, is_on_danger_zone, leave_danger_zone_flag])

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