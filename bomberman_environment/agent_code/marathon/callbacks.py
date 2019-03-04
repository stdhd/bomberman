

import numpy as np

from settings import s


def derive_state_representation(self):
    """
    From provided game_state, extract array state representation.

    Final state format specification in environment_save_states.py
    :param self:
    :return: State representation in specified format
    """

    player_block = 4+17

    state = np.zeros(self.x_y_to_index(16, 16, s.cols, s.rows) + 4 * player_block + 1)

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

                state[ind] = arena[x, y]  #  replace '17' with values from settings.py

            if explosions[x, y] != 0:

                state[ind] = -1 * 3**int(coin) * 2**explosions[x, y]

    startplayers = self.x_y_to_index(15, 15, s.cols, s.rows)  #  player blocks start here

    players.insert(0, me)

    player_ind = 0

    bomb_ind = 1

    for player in players:  # keep track of player locations and bombs
        state[startplayers + player_block * player_ind] = self.x_y_to_index(player[0], player[1], s.cols, s.rows)

        if player[3] == 0:

            player_bomb = bombs[-1 * bomb_ind]

            state[startplayers + player_block*player_ind + 2] = self.x_y_to_index(player_bomb[0], player_bomb[1],
                                                                                  s.cols, s.rows)

            state[startplayers + player_block * player_ind + 3] = player_bomb[2]

            bomb_ind += 1

        player_ind += 1

    return state







