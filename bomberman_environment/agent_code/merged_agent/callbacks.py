import numpy as np
import warnings
from random import shuffle
from state_functions.indices import *
from state_functions.state_representation import *
from Q.Oktett_Training import *
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
    self.epsilon = 0.2
    self.train_flag = True
    self.obs_radius = 0
    self.obs_object = ObservationObject(self.obs_radius, ["d_closest_coin_dir", "d_closest_crate_dir", "d_closest_safe_field_dir", "d4_is_safe_to_move_a_l",
                                        "d4_is_safe_to_move_b_r", "d4_is_safe_to_move_c_u", "d4_is_safe_to_move_d_d", "me_has_bomb"], self.logger)
    # Used for plotting
    self.total_steps_over_episodes = 0
    self.total_deaths_over_episodes = 0
    self.number_of_episode = 0

    observation_size = self.obs_object.get_observation_size()
    filename = self.obs_object.get_file_name_string()

    
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'] containing their learned rewards
    try:
        self.q_table = np.load(os.path.join('agent_code', 'merged_agent', 'q_table-' + filename + '.npy'))
        self.logger.debug('LOADED Q')
        if self.q_table.shape[1] != 6:
            raise Exception('q_table size does not fit') 
    except Exception as e:
        self.q_table = np.empty([0,6])
        self.logger.info(f'OVERWRITTEN: {e}')

    # observation_db contains learned states
    try:
        self.observation_db = np.load(os.path.join('agent_code', 'merged_agent', 'observation-' + filename + '.npy'))
        self.logger.debug('LOADED Obs')
    except:
        self.observation_db = np.empty([0, observation_size])
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
    self.obs_object.set_state(derive_state_representation(self))
    observation = self.obs_object.create_observation(np.array([int(0)]))[0]
    
    # self.logger.info(f'BOMBS: {bombs}')
    self.logger.info(f'self: {[x, y]}')
    self.logger.info(f'Observation: {observation}')
    
# ------------------------ Testing agent ------------------------
    if not self.train_flag:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
        if observation_ind.shape[0] != 0:
            self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))
        else:
            # TODO: regression
            self.last_action_ind = np.random.randint(0,6)
# ------------------------ Training agent ------------------------
    else:
        # Search for state in observation_db
        # If/else needed because np.where can only be done if self.observation_db is not empty
        if self.observation_db.shape[0] != 0:
            warnings.simplefilter(action='ignore', category=FutureWarning)
            observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
            # self.logger.info(f'OBSERVATIONS_IND: {observation_ind}')
        else:
            observation_ind = np.array([])

        # For updating only the original observation is needed as its rotations have the same reward.
        self.last_observation = observation

        observations, self.last_action_rotations = get_transformations(observation, self.obs_object.radius,
                                                        self.obs_object.get_direction_sensitivity())
        # self.logger.info(f'DIRECTION SENS: {self.obs_object.get_direction_sensitivity()}')
        self.last_q_ind = []
        # self.logger.info(f'OBSERVATIONS: {observations}')
        # Choose random action and if current observation is unknown add it and its rotations to observation_db
        if self.epsilon > np.random.uniform(0,1):
            self.last_action_ind = np.random.randint(0,6)
            # self.logger.info(f'RANDOM ACTION: {observation_ind.shape[0]}')
            if observation_ind.shape[0] == 0:
                # observations = np.unique(observations, axis=0)
                for obs in observations: 
                    obs_ind = np.where((self.observation_db == obs).all(axis=1))[0]
                    if obs_ind.shape[0] == 0: # test for uniqueness
                        self.observation_db = np.append(self.observation_db, np.array([obs]), axis = 0)
                        self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
                        self.last_q_ind.append(self.q_table.shape[0] - 1)
                    else:
                        self.last_q_ind.append(obs_ind[0])
            else:
                for obs in observations: 
                    self.last_q_ind.append(np.where((self.observation_db == obs).all(axis=1))[0][0])
        else:
            # If observation is unknown it and its rotations have to be added to observation_db and a random action is chosen.
            if observation_ind.shape[0] == 0:
                self.last_action_ind = np.random.randint(0,6)
                # observations, indices = np.unique(observations, axis=0, return_index=True)
                for obs in observations: 
                    obs_ind = np.where((self.observation_db == obs).all(axis=1))[0]
                    # self.logger.info(f'OBS_IND NOT RANDOM: {obs_ind}')
                    if obs_ind.shape[0] == 0: # test for uniqueness
                        self.observation_db = np.append(self.observation_db, np.array([obs]), axis = 0)
                        self.q_table = np.append(self.q_table, np.zeros([1, self.q_table.shape[1]]), axis = 0)
                        self.last_q_ind.append(self.q_table.shape[0] - 1)
                    else:
                        self.last_q_ind.append(obs_ind[0])
            # If observation is known the action with the highest value is chosen and observations indices are searched for rewarding
            elif observation_ind.shape[0] != 0:
                # self.logger.info(f'Q-TABLE: {self.q_table[observation_ind[0]]}')
                self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))
                for obs in observations: 
                    # self.logger.info(f'NP WHERE: {np.where((self.observation_db == obs).all(axis=1))}')
                    self.last_q_ind.append(np.where((self.observation_db == obs).all(axis=1))[0][0])
    # self.logger.info(f'self.last_q_ind: {self.last_q_ind}')
    self.next_action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][self.last_action_ind]

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    # self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    # self.logger.info(f'Events: {self.events}')
    # Find new observation index to get its best action value for updating 
    self.obs_object.set_state(derive_state_representation(self))
    observation = self.obs_object.create_observation(np.array([0]))[0]
    warnings.simplefilter(action='ignore', category=FutureWarning)
    observation_ind = np.where((self.observation_db == observation).all(axis=1))[0]
    if observation_ind.shape[0] == 0:
        current_best_value = 0
    else:
        current_best_value = self.q_table[observation_ind].max()
    
    # Get reward only for the originial observation. The reward for its rotations is the same
    reward = _getReward(self.obs_object, self.events, self.last_observation, self.logger)
    # self.logger.info(f'REWARD: {reward}')
    for ind, rotation in enumerate(self.last_action_rotations):
        # self.logger.info(f'last_action_rotations: {self.last_action_rotations}')
        # self.logger.info(f'last_action_ind : {self.last_action_ind}')
        # self.logger.info(f'rotation : {rotation}')
        # self.logger.info(f'Q_ind: {self.last_q_ind}')
        self.logger.info(f'Q-TABLE BEFORE: {self.q_table[self.last_q_ind[ind]]}')
        # self.logger.info(f'INDEXING: {int(np.where(rotation == self.last_action_ind)[0][0])}')
        self.q_table[self.last_q_ind[ind], int(np.where(rotation == self.last_action_ind)[0][0])] = (1-self.learning_rate) * self.q_table[self.last_q_ind[ind], int(np.where(rotation == self.last_action_ind)[0][0])] \
                                                                + self.learning_rate * (reward + self.discount * current_best_value)
        self.logger.info(f'Q-TABLE UPDATED: {self.q_table[self.last_q_ind[ind]]}')

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    # Do the same as in reward_update
    reward_update(self)
    filename = self.obs_object.get_file_name_string()
    np.save(os.path.join('agent_code', 'merged_agent', 'observation-' + filename), self.observation_db)
    np.save(os.path.join('agent_code', 'merged_agent', 'q_table-' + filename), self.q_table)
    self.total_steps_over_episodes += self.game_state['step']
    if 13 in self.events or 14 in self.events: self.total_deaths_over_episodes += 1
    self.number_of_episode += 1
    if self.number_of_episode % 50 == 0: 
        self.logger.info(f'Episode number, Total Steps and Deaths: {self.number_of_episode, self.total_steps_over_episodes, self.total_deaths_over_episodes}')
        self.total_steps_over_episodes, self.total_deaths_over_episodes = 0, 0

def _getReward(obs_object, events, old_observation, logger):
    reward = 0
    ccdir_feature_ind = obs_object.get_feature_index("d_closest_coin_dir")
    ccrdir_feature_ind = obs_object.get_feature_index("d_closest_crate_dir")
    csfdir_feature_ind = obs_object.get_feature_index("d_closest_safe_field_dir")
    ismal_feature_ind = obs_object.get_feature_index("d4_is_safe_to_move_a_l")
    ismbr_feature_ind = obs_object.get_feature_index("d4_is_safe_to_move_b_r")
    ismcu_feature_ind = obs_object.get_feature_index("d4_is_safe_to_move_c_u")
    ismdd_feature_ind = obs_object.get_feature_index("d4_is_safe_to_move_d_d")
    logger.info(f'ISMAL: {ismal_feature_ind}')
    logger.info(f'CCDIR: {ccdir_feature_ind}')
    if 0 in events: # Left
        if csfdir_feature_ind != None and old_observation[csfdir_feature_ind] == 0: reward += 800 # Reward when agent chooses direction safe field
        if ismal_feature_ind != None and csfdir_feature_ind != None \
            and old_observation[ismal_feature_ind] == 0 and old_observation[csfdir_feature_ind] > 3: reward -= 300 # Punish when agent chooses direction to danger zone, explosion or invalid action
        if ccrdir_feature_ind != None and ismal_feature_ind != None \
            and old_observation[ccrdir_feature_ind] == 0 and old_observation[ismal_feature_ind] == 1: reward += 400 # Reward if following closest crate feature
        if ccdir_feature_ind != None and ismal_feature_ind != None \
            and old_observation[ccdir_feature_ind] == 0 and old_observation[ismal_feature_ind] == 1: reward += 300 # Reward if following closest coin feature
        reward -= 50
    elif 1 in events: # Right
        if csfdir_feature_ind != None and old_observation[csfdir_feature_ind] == 1: reward += 800
        if ismbr_feature_ind != None and csfdir_feature_ind != None \
            and old_observation[ismbr_feature_ind] == 0 and old_observation[csfdir_feature_ind] > 3: reward -= 300
        if ccrdir_feature_ind != None and ismbr_feature_ind != None \
            and old_observation[ccrdir_feature_ind] == 1 and old_observation[ismbr_feature_ind] == 1: reward += 400
        if ccdir_feature_ind != None and ismbr_feature_ind != None \
            and old_observation[ccdir_feature_ind] == 1 and old_observation[ismbr_feature_ind] == 1: reward += 300
        reward -= 50
    elif 2 in events: # Up
        if csfdir_feature_ind != None and old_observation[csfdir_feature_ind] == 2: reward += 800
        if ismcu_feature_ind != None and csfdir_feature_ind != None \
            and old_observation[ismcu_feature_ind] == 0 and old_observation[csfdir_feature_ind] > 3: reward -= 300
        if ccrdir_feature_ind != None and ismcu_feature_ind != None \
            and old_observation[ccrdir_feature_ind] == 2 and old_observation[ismcu_feature_ind] == 1: reward += 400
        if ccdir_feature_ind != None and ismcu_feature_ind != None \
            and old_observation[ccdir_feature_ind] == 2 and old_observation[ismcu_feature_ind] == 1: reward += 300
        reward -= 50
    elif 3 in events: # Down
        if csfdir_feature_ind != None and old_observation[csfdir_feature_ind] == 3: reward += 800
        if ismdd_feature_ind != None and csfdir_feature_ind != None \
            and old_observation[ismdd_feature_ind] == 0 and old_observation[csfdir_feature_ind] > 3: reward -= 300
        if ccrdir_feature_ind != None and ismdd_feature_ind != None \
            and old_observation[ccrdir_feature_ind] == 3 and old_observation[ismdd_feature_ind] == 1: reward += 400
        if ccdir_feature_ind != None and ismdd_feature_ind != None \
            and old_observation[ccdir_feature_ind] == 3 and old_observation[ismdd_feature_ind] == 1: reward += 300
        reward -= 50
    if 4 in events: # Wait
        if ismal_feature_ind != None and ismbr_feature_ind != None and ismcu_feature_ind != None and ismdd_feature_ind != None and csfdir_feature_ind != None\
            and old_observation[ismal_feature_ind] == 0 and old_observation[ismbr_feature_ind] == 0 \
            and old_observation[ismcu_feature_ind] == 0 and old_observation[ismdd_feature_ind] == 0 \
            and old_observation[csfdir_feature_ind] > 3:
            reward += 500
        else:
            reward -= 500
    if 5 in events: reward -= 0 # Interrupted 
    if 6 in events: reward -= 1000 # Invalid action
    if 7 in events: # Bomb dropped
        if ccrdir_feature_ind != None and old_observation[ccrdir_feature_ind] == 5: reward += 600 # Reward when agent sets bomb before crate
        if ccrdir_feature_ind != None and old_observation[ccrdir_feature_ind] == 4: reward -= 300
        reward -= 100
    if 8 in events: reward += 0 # Bomb exploded
    if 9 in events: reward += 0 # Crate destroyed
    if 10 in events: reward += 0 # Coin found
    if 11 in events: reward += 500 # Coin collected
    if 12 in events: reward += 500 # Killed opponent
    if 13 in events: reward -= 1000 # Killed self
    if 14 in events: reward -= 0 # Got killed either by himself or by enemy
    if 15 in events: reward -= 0 # Opponent eliminated
    if 16 in events: reward -= 0 # Survived round
        
    return reward

# def _derive_state_representation(self, where):
#     """
#     From provided game_state, extract array state representation. Use this when playing game (not training)

#     Final state format specification in environment_save_states.py
#     :param self:
#     :return: State representation in specified format
#     """
#     player_block = 4+17
#     state = np.zeros(indices.x_y_to_index(s.cols - 2, s.rows - 2, s.cols, s.rows) + 4 * player_block + 1)
#     state[-1] = self.game_state['step']
#     arena = self.game_state['arena']
#     explosions = self.game_state['explosions']
#     coin_locs = self.game_state['coin_locs']
#     players = self.game_state['others']
#     bombs = self.game_state['bombs']
#     me = self.game_state['self']
#     for x, y in coin_locs:
#         ind = indices.x_y_to_index(x, y, s.cols, s.rows) - 1
#         state[ind] = 3

#     for x in range(arena.shape[0] ):
#         if x == 0 or arena.shape[0] - 1:
#             continue
#         for y in range(arena.shape[1]):
#             if y == 0 or arena.shape[1] - 1 or (x + 1) * (y + 1) % 2 == 1:
#                 continue
#             ind = indices.x_y_to_index(x, y, s.cols, s.rows) - 1
#             coin = state[ind] == 3
#             if not coin:
#                 state[ind] = arena[x, y]  # replace '17' with values from settings.py
#             if explosions[x, y] != 0:
#                 state[ind] = -1 * 3**int(coin) * 2**explosions[x, y]

#     startplayers = indices.x_y_to_index(15, 15, s.cols, s.rows)  # player blocks start here
#     # self.logger.info(f'PLAYER BEFORE: {players, where}')
#     # Strange behaviour of game_state('others') which returns self (plus others) as others when called from act() except for the first step 
#     # where only others are returned for both methods
#     if me not in players: 
#         players.insert(0, me)
#     player_ind = 0
#     bomb_ind = 0
#     # self.logger.info(f'PLAYER AFTER: {players, where}')
#     #self.logger.info(f'PLAYER: {players}')
#     for player in players:  # keep track of player locations and bombs
#         state[startplayers + player_block * player_ind] = indices.x_y_to_index(player[0], player[1], s.cols, s.rows)
#         if player[3] == 0:
#             player_bomb = bombs[bomb_ind]  # count through bombs and assign a dropped bomb to each player
#             # who is not holding a bomb
#             state[startplayers + player_block*player_ind + 2] = indices.x_y_to_index(player_bomb[0], player_bomb[1],
#                                                                                      s.cols, s.rows)
#             state[startplayers + player_block * player_ind + 3] = player_bomb[2]  # bomb timer
#             bomb_ind += 1
#         player_ind += 1
#     return state