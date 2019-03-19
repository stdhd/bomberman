import numpy as np
import warnings
from state_functions.indices import *
from state_functions.state_representation import *
from agent_code.observation_object import ObservationObject
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
    self.epsilon = -1
    self.train_flag = False
    self.obs_object = ObservationObject(1, ['d_closest_coin_dir', 'd_closest_safe_field_dirNEW', 'me_has_bomb'], None)
    # Used for plotting
    self.total_steps_over_episodes = 0
    self.total_deaths_over_episodes = 0
    self.number_of_episode = 0

    observation_size = self.obs_object.get_observation_size() - 1 #cut out field radius
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    filename = self.obs_object.get_file_name_string()

    try:
        self.q_table = np.load(os.path.join('data', 'qtables', filename, 'q_table-' + filename + '.npy'))
        self.quantities = np.load(os.path.join('data', 'qtables', filename, 'quantity-' + filename + '.npy'))
        self.logger.debug('LOADED Q')
        if self.q_table.shape[1] != 6:
            raise Exception('q_table size does not fit') 
    except Exception as e:
        self.q_table = np.empty([0,6])
        self.logger.info(f'OVERWRITTEN: {e}')

    # Zx10 array with 3x3 observation around agent plus coin_flag
    try:
        self.observation_db = np.load(os.path.join('data', 'qtables', filename, 'observation-' + filename + '.npy'))
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
    bombs = self.game_state['bombs']
    # self.logger.info(f'BOMBS: {bombs}')
    rep = derive_state_representation(self)  # DEBUG
    self.obs_object.set_state(rep)
    observation = self.obs_object.create_observation(np.array([0]))[0]
    # observation = np.delete(observation, [0])
    self.old_observation = observation
    # self.logger.info(f'self: {[x, y]}')
    # self.logger.info(f'Observation: {observation}')
    

    # Search for state in observation_db
    # If/else needed because np.where can only be done if self.observation_db is not empty
    if self.observation_db.shape[0] != 0:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        observation_ind = np.where((self.observation_db == np.array([observation])).all(axis=1))[0]
       # np.where((KNOWN == np.array([observation])).all(axis=1))[0]
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
        # If training mode is on and observation_db has no entry it has to be appended and a random action is chosen.
        if observation_ind.shape[0] == 0 and self.train_flag:
            self.last_action_ind = np.random.randint(0,6)
            self.observation_db = np.append(self.observation_db, np.array([observation]), axis = 0)
            self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
            self.last_q_ind = self.q_table.shape[0] - 1
        # If observation_db has entry the action with the highest value is chosen.
        elif observation_ind.shape[0] != 0:
            print("---")
            print("KNOWN observation")
            print(self.observation_db[observation_ind[0]])
            # print(self.observation_db[observation_ind])
            print('LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB')
            print(self.q_table[observation_ind[0]])
            print("Quantities: ")
            print(self.quantities[observation_ind[0]])
            self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))

        # If test mode and observation_db has no entry
        else:
            # TODO: regression
            print("unknown observation")
            self.last_action_ind = np.random.randint(0,6)

    self.next_action = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'][self.last_action_ind]
   # if self.total_steps_over_episodes == 0:
    #    self.next_action = 'UP'
    print(self.next_action)

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.total_steps_over_episodes += self.game_state['step']
    if 13 in self.events or 14 in self.events: self.total_deaths_over_episodes += 1
    self.number_of_episode +=1
    if self.number_of_episode % 250 == 0: 
        self.logger.info(f'Episode number, Total Steps and Deaths: {self.number_of_episode, self.total_steps_over_episodes, self.total_deaths_over_episodes}')
        self.total_steps_over_episodes, self.total_deaths_over_episodes = 0, 0