
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    self.q_table = np.empty([0,6])
    # Zx10 array with 3x3 observation around agent plus coin_flag
    self.observation_db = np.empty([0,10])
    self.learning_rate = 0.4
    self.discount = 0.7
    self.epsilon = 0.9


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
    # Default action
    self.next_action = 'RIGHT'

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    # bombs = self.game_state['bombs']
    # bomb_xys = [(x,y) for (x,y,t) in bombs]
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    # bomb_map = np.ones(arena.shape) * 5
    # for xb,yb,t in bombs:
    #     for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
    #         if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #             bomb_map[i,j] = min(bomb_map[i,j], t)
    
    agent_coords = np.array([x, y])
    nearest_coin_coords = coins[np.argmin(np.sum(np.abs(agent_coords - coins), axis=1))]
    self.logger.info(f'self: {agent_coords}')
    self.logger.info(f'coin: {nearest_coin_coords}')
    coin_flag = ""
    if nearest_coin_coords[0] > x:
        coin_flag = "r"
    elif nearest_coin_coords[0] < x:
        coin_flag = "l"

    if nearest_coin_coords[1] > y:
        coin_flag = coin_flag + "d"
    elif nearest_coin_coords[1] < y:
        coin_flag = coin_flag + "u"

    observation = np.array([arena[x-1, y-1], arena[x, y-1], arena[x+1, y-1], arena[x-1, y], arena[x, y], arena[x+1, y], arena[x-1, y+1], arena[x, y+1], arena[x+1, y+1], coin_flag])
    self.logger.info(f'Observation: {observation}')
    # self.logger.info(f'Observation db: {self.observation_db}')

    # If/else needed because np.where can only be done if self.observation_db is not empty
    if self.observation_db.shape[0] != 0:
        observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
        self.last_q_ind = observation_ind
    else:
        observation_ind = np.array([])
    
    if observation_ind.shape[0] == 0:
        self.observation_db = np.append(self.observation_db, np.array([observation]), axis = 0)
        self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
        self.last_action_ind = np.random.randint(1,7)
        self.last_q_ind = self.q_table.shape[0] - 1
    else:
        self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind] == self.q_table[observation_ind].max()))

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
    self.logger.info(f'Events: {type(self.events)}')
    reward = 0
    # Left, right, up, down, waited (0-4)
    if 0 in self.events:
        reward -= 1
        action_index = 2
    if 1 in self.events:
        reward -= 1
        action_index = 3
    if 2 in self.events:
        reward -= 1
        action_index = 0
    if 3 in self.events:
        reward -= 1
        action_index = 1
    if 4 in self.events:
        reward -= 5
        action_index = 5
    # Bomb dropped
    if 7 in self.events:
        reward -= 10
        action_index = 4
    # Coin found
    if 10 in self.events:
        reward += 50

    # TODO: Find my_best_value from new state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']

    # self.last_action_ind not really needed as it can be concluded from self.events
    self.q_table[self.last_q_ind, self.last_action_ind] = (1-self.learning_rate) * self.q_table[self.last_q_ind, self.last_action_ind]
                                                                + self.learning_rate * (reward + self.discount * my_best_value)

    # Q-table mit 1-Feld Radius an Zuständen, also 9er Vektor:
        # -1 = Wall
        #  0 = Frei
        #  3 = Coin

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
