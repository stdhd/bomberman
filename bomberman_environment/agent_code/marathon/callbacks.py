

import numpy as np

from settings import s

from observation import create_observation, find_equivalent

from agent_code.marathon.kernel_regression import *


def setup(self):
    """
    Initialize agent using data from q tables.
    :param self:
    :param q_tables: filepath to
    :return:
    """

    self.isInitialized = False

    q_table_path = 'agent_code/marathon/tables.npy'  # load q table from RL

    obs_action_path = 'agent_code/marathon/obs_actions.npy'  # load dataset of observation+action tuples

    kernel_alpha_path = 'agent_code/marathon/alpha.npy'  # load precomputed kernel 'alpha' vector

    q_table = np.load(q_table_path)

    self.obs = q_table[0]

    self.q = q_table[1]

    self.OBS_ACTIONS = np.load(obs_action_path)

    self.alpha = np.load(kernel_alpha_path)

    self.sig, self.tau = None, None # FIXME

    self.isInitialized = True


def act(self):
    """
    Derives state representation, calls observation function, and selects next action from q table.
    :param self:
    :return:
    """

    self.next_action = 'WAIT'

    try:
        if not self.isInitialized:
            self.logger.error("act: Initialization failed before action. ")
            setup(self)
            if not self.isInitialized:
                self.logger.error("act: Initialization failed!")
                raise RuntimeError
    except AttributeError:
        err = "act: Setup was not called before act(self). "
        self.logger.error(err)
        raise RuntimeError(err)

    state = derive_state_representation(self)

    observation = create_observation(state, self.radius, [0])

    obs_index = find_equivalent(observation, self.obs)

    if obs_index == -1:  # state not found, use regression

        choice = regression_action(self, observation)

    else:

        choice = np.random.choice(np.flatnonzero(self.q[obs_index] == self.q[obs_index].max()))  # choose highest scoring
        # action from q table

    self.next_action = self.ACTIONS[choice]


def regression_action(self, observation):
    """
    Kernel regression to get highest scoring action from observation.
    :param self:
    :param observation:
    :return: int encoding for action (see CONVENTIONS)
    """

    obs_and_action = np.copy(observation)

    np.append(obs_and_action, 0)

    q_approx_values = np.zeros(len(s.actions))

    for i in range(len(s.actions)):

        obs_and_action[-1] = i  # find q(s, a) for this observation + action

        kernel_vects = K_sparse_pairwise(np.array([obs_and_action]), self.OBS_ACTIONS, self.sig)
        # get kernelized distances

        q_approx_values[i] = kernel_vects.dot(self.alpha)[0]  # do kernel regression

    return np.random.choice(np.flatnonzero(q_approx_values == q_approx_values.max()))


def derive_state_representation(self):
    """
    From provided game_state, extract array state representation. Use this when playing game (not training)

    Final state format specification in environment_save_states.py
    :param self:
    :return: State representation in specified format
    """

    player_block = 4+17

    state = np.zeros(self.x_y_to_index(s.cols - 2, s.rows - 2, s.cols, s.rows) + 4 * player_block + 1)

    state[-1] = self.game_state['step']

    arena = self.game_state['arena']

    explosions = self.game_state['explosions']

    coins = self.game_state['coins']

    players = self.game_state['others']

    bombs = self.game_state['bombs']

    me = self.game_state['self']

    for x, y in coins:

        ind = self.x_y_to_index(x, y, s.cols, s.rows) - 1

        state[ind] = 3

    for x in range(arena.shape[0] ):
        if x == 0 or arena.shape[0] - 1:
            continue
        for y in range(arena.shape[1]):
            if y == 0 or arena.shape[1] - 1 or (x + 1) * (y + 1) % 2 == 1:
                continue

            ind = self.x_y_to_index(x, y, s.cols, s.rows) - 1

            coin = state[ind] == 3

            if not coin:

                state[ind] = arena[x, y]  # replace '17' with values from settings.py

            if explosions[x, y] != 0:

                state[ind] = -1 * 3**int(coin) * 2**explosions[x, y]

    startplayers = self.x_y_to_index(15, 15, s.cols, s.rows)  # player blocks start here

    players.insert(0, me)

    player_ind = 0

    bomb_ind = 0

    for player in players:  # keep track of player locations and bombs
        state[startplayers + player_block * player_ind] = self.x_y_to_index(player[0], player[1], s.cols, s.rows)

        if player[3] == 0:

            player_bomb = bombs[bomb_ind]  # count through bombs and assign a dropped bomb to each player
            # who is not holding a bomb

            state[startplayers + player_block*player_ind + 2] = self.x_y_to_index(player_bomb[0], player_bomb[1],
                                                                                  s.cols, s.rows)

            state[startplayers + player_block * player_ind + 3] = player_bomb[2]  # bomb timer

            bomb_ind += 1

        player_ind += 1

    return state







