
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s
from random import *
import math



def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.coin_distance = 0
    self.obs_length = 1
    self.discount = 0.8
    self.learning_rate = 0.8
    self.epsilon = 0.1
    self.reward = 0
    self.last_observation = np.zeros([self.obs_length])
    self.last_action_index = -1
    np.random.seed()
    try:
        self.learned = np.load("learned.npy")
        self.observations = np.load("observations.npy")
    except:
        print("Error loading learned q table. using empty table instead.")
        self.learned = np.zeros([0,5])
        self.observations = np.zeros([0,self.obs_length])

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
    self.logger.info('Picking action according to rule set')

    arena = create_arena(self)
    # Gather information about the game state

    x, y, _, bombs_left, score = self.game_state['self']
    # print("x,y:" + str(x)  + "  " + str(y))

    #actions = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'])
    actions = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN'])

    current_obs = get_observation(self, arena, x, y)

    self.observations, index_current, self.learned = update_and_get_obs(self.observations, current_obs, self.learned)
    if self.epsilon > uniform(0, 1):
        choice = np.random.choice(np.arange(actions.shape[0]))

    else:
        choice = np.random.choice(np.flatnonzero(self.learned[index_current] == self.learned[index_current].max()))

    my_best_value = np.max(self.learned[index_current])

    self.observations, last_index, self.learned = update_and_get_obs(self.observations, self.last_observation, self.learned)

    if not (self.last_action_index == -1):
        tes = 123
        self.learned[last_index, self.last_action_index] = (1 - self.learning_rate) * self.learned[last_index, self.last_action_index] + self.learning_rate * (self.reward + self.discount * my_best_value)

    self.reward = 0
    self.last_action_index = choice
    self.last_observation = current_obs
    self.next_action = actions[choice]
   #  print(actions[choice])
    # print(actions[choice], self.observations, self.learned)

def create_arena(self):
    arena = self.game_state['arena']
    # print(self.game_state['bombs'])
    for x, y, time in self.game_state['bombs']:
        arena[x, y] = 9

    for x, y, t, tt, ttt in self.game_state['others']:
        arena[x, y] = 6

    for x, y in self.game_state['coins']:
        arena[x, y] = 10

    return arena

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

    reminder = [
        'MOVED_LEFT',  # 0
        'MOVED_RIGHT',
        'MOVED_UP',
        'MOVED_DOWN',
        'WAITED',  # 4
        'INTERRUPTED',
        'INVALID_ACTION',

        'BOMB_DROPPED',  # 7
        'BOMB_EXPLODED',

        'CRATE_DESTROYED',  # 9
        'COIN_FOUND',
        'COIN_COLLECTED',

        'KILLED_OPPONENT',  # 12
        'KILLED_SELF',

        'GOT_KILLED',
        'OPPONENT_ELIMINATED',
        'SURVIVED_ROUND',
    ]

    coin_collected_flag = False
    self.reward = 0
    #print(self.events)
    self.reward = -5
    for event in self.events:
        if event == 6:
            self.reward -= 10
        if event == 9:
            self.reward += 50
        if event == 11:
            self.reward += 300
            coin_collected_flag = True
        if event == 13:
            self.reward -= 1000
        if event == 14:
            self.reward -= 500

    x, y, _, bombs_left, score = self.game_state['self']
    arena = create_arena(self)
    free_tiles = np.where(arena == 0, True, False)
    targetsX, targetsY = np.where(arena == 10)
    targets = list(zip(targetsX.tolist(), targetsY.tolist()))

    (tx, ty), new_distance = look_for_targets(free_tiles, (x,y), targets, None)

    if not coin_collected_flag:
        if self.coin_distance > new_distance:
            self.reward += 5
           # print("old:" + str(self.coin_distance) + "new:" + str(new_distance) + "+5")
        elif self.coin_distance < new_distance:
            self.reward -= 5
            #print("old:" + str(self.coin_distance) + "new:" + str(new_distance) + "-5")
        else:
            self.reward -= 2

    self.coin_distance = new_distance

    self.logger.debug(f'Encountered {len(self.events)} game event(s)')




def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """

    np.save("learned.npy",self.learned)
    np.save("observations.npy", self.observations)

    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')


def get_observation(self,spielfeld, x, y):
    '''
    TODO: change x,y, arena array[x,y]
    TODO: Flexible radius
    :param spielfeld:
    :param y:
    :param x:
    :return:
    '''
    observation = np.zeros([self.obs_length])
    free_tiles = np.where(spielfeld != -1, True, False)
    start = (x,y)
    targetsX, targetsY = np.where(spielfeld == 10)
    tmp = 0
    if targetsX.shape[0] > 0:
        targets = list(zip(targetsX.tolist(), targetsY.tolist()))


        # print(next_target)
        (tx,ty), self.coin_distance = look_for_targets(free_tiles, start, targets, None)
        #print("---")
        #print(tx, ty)
        #print(x,y)

        # print(self.coin_distance)

        if (tx,ty) == (x+1,y):
            # move right
            tmp = 1
        if (tx,ty) == (x-1,y):
            # move left
            tmp = 2
        if (tx,ty) == (x,y+1):
            # move up
            tmp = 3
        if (tx,ty) == (x,y-1):
            # move down
            tmp = 4

    observation[0] = tmp

    # k = 1
    # radius = 3
    # obs = np.zeros([2*radius+1,2*radius+1])
    # for i in range(2*radius+1):
    #     for j in range(2*radius+1):
    #         observation[k] = get_spielfeld(x-radius + j, y-radius + i,spielfeld)
    #         k += 1

    return observation

def get_spielfeld(x,y,spielfeld):
    if y > spielfeld.shape[1] - 1 or x > spielfeld.shape[0] - 1 or y < 0 or x < 0:
        #print("collision")
        return - 1
    else:
        return spielfeld[x,y]

def update_and_get_obs(db, new_obs, learned):
    '''
    TODO: Flexible radius
    TODO: Find transformed observations
    :param db:
    :param new_obs:
    :param learned:
    :return:
    '''
    temp = -1
    for i in range(db.shape[0]):
        if np.array_equal(db[i], new_obs):
            return db, i, learned
    db = np.append(db,np.array([new_obs]), axis = 0)
    learned = np.append(learned, np.zeros([1,learned.shape[1]]), axis = 0)
    return db, (db.shape[0] - 1), learned


def look_for_targets(free_space, start, targets, logger=None):
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
        if parent_dict[current] == start:
            # print(best)
            return current, best_dist
        current = parent_dict[current]
