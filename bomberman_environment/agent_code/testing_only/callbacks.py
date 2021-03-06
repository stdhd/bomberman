import numpy as np
import warnings
from state_functions.indices import *
from state_functions.state_representation import *
from agent_code.observation_object import ObservationObject
import os
import pickle

from settings import s

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    # self.logger.setLevel("DEBUG")
    self.train_flag = False
    self.obs_object = ObservationObject(0, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                'me_has_bomb',
                                'dead_end_detect',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d',
                                'd_best_bomb_dropping_dir',
                                'd_closest_enemy_dir'
                                ], None)

    self.clf = None

    with open("data/qtables/" + self.obs_object.get_file_name_string() + "/3.dt.p", "rb") as f:
        self.clf = pickle.load(f)  # FIXME load regression model
        self.CLASSIFIER = True  #  Indicates whether we are using a classifier or a regressor

    self.FORCE_REGRESSION = True  # FIXME

    self.logger.debug("Called setup")
    # Used for plotting
    self.total_steps_over_episodes = 0
    self.total_deaths_over_episodes = 0
    self.number_of_episode = 0

    observation_size = self.obs_object.get_observation_size() - 1 #cut out field radius
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    filename = self.obs_object.get_file_name_string()

    try:
        self.logger.info("Loading Q tables and quantities for: " + filename)
        self.q_table = np.load(os.path.join('data', 'qtables', filename, 'q_table-' + filename + '.npy'))
        self.quantities = np.load(os.path.join('data', 'qtables', filename, 'quantity-' + filename + '.npy'))
        self.logger.info('LOADED Q')
        if self.q_table.shape[1] != 6:
            raise Exception('q_table size does not fit')
    except Exception as e:
        self.q_table = np.empty([0, 6])
        self.logger.info(f'OVERWRITTEN: {e}')

    # Zx10 array with 3x3 observation around agent plus coin_flag
    try:
        self.observation_db = np.load(os.path.join('data', 'qtables', filename, 'observation-' + filename + '.npy'))
        self.logger.info('LOADED Obs')
    except:
        self.observation_db = np.empty([0,observation_size])
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

    # Gather information about the game state
    x, y, _, bombs_left, score = self.game_state['self']
    rep = derive_state_representation(self)
    self.obs_object.set_state(rep)
    observation = self.obs_object.create_observation(np.array([0]))[0]
    self.old_observation = observation
    observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
    # If observation_db has entry the action with the highest value is chosen.
    if not self.FORCE_REGRESSION and observation_ind.shape[0] != 0:
        self.logger.debug("KNOWN observation: ")
        self.logger.debug("\n"+str(self.observation_db[observation_ind[0]][:self.obs_object.window_size**2].reshape(
            (self.obs_object.window_size, self.obs_object.window_size)).T))
        self.logger.debug("Features: " + str(self.observation_db[observation_ind[0]][self.obs_object.window_size**2:]))
        self.logger.debug(str(('LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB')))
        self.logger.debug(self.q_table[observation_ind[0]])
        self.logger.debug("Quantities: " + str(self.quantities[observation_ind[0]]))
        self.last_action_ind = np.random.choice(
            np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))


    # If test mode and observation_db has no entry
    else:
        # TODO: regression
        self.logger.debug("\n"+str(observation[:self.obs_object.window_size**2].reshape(
            (self.obs_object.window_size, self.obs_object.window_size)).T))
        self.logger.debug("Features: " + str(observation[self.obs_object.window_size**2:]))
        self.logger.debug(str(('LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB')))
        self.last_action_ind = np.random.randint(0, 6)
        best_action_score = -10000
        if not self.CLASSIFIER:
            print("Using regression")
            for i in range(6):
                try_obs = np.append(np.copy(observation), i)
                try_obs = try_obs[np.newaxis, :]
                value = self.clf.predict(try_obs)

                print("Action score for", i, value)

                if value >= best_action_score:
                    best_action_score = value
                    self.last_action_ind = i
        else:
            print("Using classification")
            try_obs = np.copy(observation)[np.newaxis, :]
            self.last_action_ind = self.clf.predict(try_obs)[0]
        print("Chose action", self.last_action_ind)

    if not self.FORCE_REGRESSION:
        # Deadlock detection:
        self.last_visited = np.append(self.last_visited, np.array([[x, y]]), axis=0)
        if (self.last_visited[-1] == self.last_visited[-3]).all() and (
                self.last_visited[-2] == self.last_visited[-4]).all() \
                and (self.last_visited[-1] != self.last_visited[-2]).all():

            # Case: Q Table has entry for observation
            if observation_ind.shape[0] != 0:
                self.logger.info("DEADLOCK DETECTED. DO " + str(self.repeated_deadlock) + " BEST ALTERNATIVE NOW")
                print("DEADLOCK DETECTED. DO " + str(self.repeated_deadlock) + " BEST ALTERNATIVE NOW")
                alternatives = np.argsort(self.q_table[observation_ind[0]])
                self.last_action_ind = alternatives[-np.min(np.array([self.repeated_deadlock, 3]))]

            # Case: No Q Table entry, so random action is chosen
            else:
                self.last_action_ind = np.random.choice(np.array([[0, 1, 2, 3]]))
            self.repeated_deadlock += 1

        # Case: No deadlock -> reset
        else:
            self.repeated_deadlock = 1

    self.next_action = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'][self.last_action_ind]
    self.logger.info("Next action: " + self.next_action)


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