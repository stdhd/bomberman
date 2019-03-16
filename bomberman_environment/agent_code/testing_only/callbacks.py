import numpy as np
import warnings
from random import shuffle
from agent_code.merged_agent import indices
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
    self.obs_object = ObservationObject(1, self.logger, ['d_closest_coin_dir'])
    # Used for plotting
    self.total_steps_over_episodes = 0
    self.total_deaths_over_episodes = 0
    self.number_of_episode = 0

    observation_size = self.obs_object.get_observation_size() - 1 #cut out field radius
    # Zx6 array with actions ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    filename = self.obs_object.get_file_name_string()

    try:
        self.q_table = np.load(os.path.join('data', 'qtables', filename, 'q_table-' + filename + '.npy'))
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
    self.obs_object.set_state(derive_state_representation(self, "ACT"))
    observation = self.obs_object.create_observation(np.array([int(0)]))[0]
    print(observation)
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
            print("KNOWN observation")
            # print(self.observation_db[observation_ind[0]])
            # print(self.observation_db[observation_ind])
            self.last_action_ind = np.random.choice(np.flatnonzero(self.q_table[observation_ind[0]] == self.q_table[observation_ind[0]].max()))
            # print(self.q_table[observation_ind[0]])
        # If test mode and observation_db has no entry
        else:
            # TODO: regression
            print("unknown observation")
            self.last_action_ind = np.random.randint(0,6)

    self.next_action = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'][self.last_action_ind]
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


def derive_state_representation(self, where):
    """
    From provided game_state, extract array state representation. Use this when playing game (not training)

    Final state format specification in environment_save_states.py
    :param self:
    :return: State representation in specified format
    """
    player_block = 4+17
    state = np.zeros(indices.x_y_to_index(s.cols - 2, s.rows - 2, s.cols, s.rows) + 4 * player_block + 1)
    state[-1] = self.game_state['step']
    arena = self.game_state['arena']
    explosions = self.game_state['explosions']
    coins = self.game_state['coins']
    players = self.game_state['others']
    bombs = self.game_state['bombs']
    me = self.game_state['self']
    for x, y in coins:
        ind = indices.x_y_to_index(x, y, s.cols, s.rows) - 1
        state[ind] = 3

    for x in range(arena.shape[0] ):
        if x == 0 or arena.shape[0] - 1:
            continue
        for y in range(arena.shape[1]):
            if y == 0 or arena.shape[1] - 1 or (x + 1) * (y + 1) % 2 == 1:
                continue
            ind = indices.x_y_to_index(x, y, s.cols, s.rows) - 1
            coin = state[ind] == 3
            if not coin:
                state[ind] = arena[x, y]  # replace '17' with values from settings.py
            if explosions[x, y] != 0:
                state[ind] = -1 * 3**int(coin) * 2**explosions[x, y]

    startplayers = indices.x_y_to_index(15, 15, s.cols, s.rows)  # player blocks start here
    # self.logger.info(f'PLAYER BEFORE: {players, where}')
    # Strange behaviour of game_state('others') which returns self (plus others) as others when called from act() except for the first step 
    # where only others are returned for both methods
    if me not in players: 
        players.insert(0, me)
    player_ind = 0
    bomb_ind = 0
    # self.logger.info(f'PLAYER AFTER: {players, where}')
    #self.logger.info(f'PLAYER: {players}')
    for player in players:  # keep track of player locations and bombs
        state[startplayers + player_block * player_ind] = indices.x_y_to_index(player[0], player[1], s.cols, s.rows)
        if player[3] == 0:
            player_bomb = bombs[bomb_ind]  # count through bombs and assign a dropped bomb to each player
            # who is not holding a bomb
            state[startplayers + player_block*player_ind + 2] = indices.x_y_to_index(player_bomb[0], player_bomb[1],
                                                                                     s.cols, s.rows)
            state[startplayers + player_block * player_ind + 3] = player_bomb[2]  # bomb timer
            bomb_ind += 1
        player_ind += 1
    return state