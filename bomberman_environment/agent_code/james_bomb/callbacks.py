import numpy as np
import warnings
from agent_code.james_bomb.helper_functions import *
from agent_code.james_bomb.observation_object import ObservationObject
import os
import pickle

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    # self.logger.setLevel("DEBUG")
    self.obs_object = ObservationObject(-1, ['d_closest_coin_dir',
                                                'd_closest_safe_field_dir',
                                                'me_has_bomb',
                                                'dead_end_detect',
                                                'd4_is_safe_to_move_a_l',
                                                'd4_is_safe_to_move_b_r',
                                                'd4_is_safe_to_move_c_u',
                                                'd4_is_safe_to_move_d_d',
                                                'd_best_bomb_dropping_dir',
                                                'd_closest_enemy_dir', 
                                                'enemy_in_bomb_area'
                                                ], None)


    observation_size = self.obs_object.get_observation_size() - 1 #cut out field radius
    filename = self.obs_object.get_file_name_string()

    try:
        # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        self.q_table = np.load(os.path.join('agent_code', 'james_bomb', 'q_table-' + filename + '.npy'))
        if self.logger: self.logger.info('LOADED Q')
        if self.q_table.shape[1] != 6:
            raise Exception('q_table size does not fit')
    except Exception as e:
        self.q_table = np.empty([0, 6])
        self.logger.info(f'Q OVERWRITTEN: {e}')

    try:
        self.observation_db = np.load(os.path.join('agent_code', 'james_bomb', 'observation-' + filename + '.npy'))
        self.logger.info('LOADED OBS')
    except Exception as e:
        self.observation_db = np.empty([0,observation_size])
        self.logger.info(f'OBS OVERWRITTEN: {e}')
        if self.observation_db.shape[1] != observation_size:
            raise Exception('observation_db size does not fit')

    self.repeated_deadlock = 1
    self.last_visited = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

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

    self.obs_object.set_state(derive_state_representation(self))
    observation = self.obs_object.create_observation(np.array([0]))[0]
    self.old_observation = observation

    # warnings.simplefilter(action='ignore', category=FutureWarning)
    observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]


    # If observation_db has entry the action with the highest value is chosen.
    if observation_ind.shape[0] != 0:
        self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))
    # If observation_db has no entry
    else:
        # Random action
        self.last_action_ind = np.random.randint(0,6)

    self.next_action = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'][self.last_action_ind]


def reward_update(self):
    """
    Not needed because this is testing only
    """

def end_of_episode(self):
    """
    Not needed because this is testing only
    """