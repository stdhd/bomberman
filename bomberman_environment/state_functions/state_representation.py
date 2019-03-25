from state_functions.indices import *


def derive_state_representation(self):
    """
    From provided game_state, extract array state representation. Use this when playing game (not training)

    Final state format specification in Google Drive
    :param self:
    :return: State representation in specified format
    """
    player_block = 4 + 17
    state = np.zeros(x_y_to_index(s.cols - 2, s.rows - 2) + 4 * player_block + 1)
    state[-1] = self.game_state['step']
    arena = self.game_state['arena']
    explosions = self.game_state['explosions']
    coins = self.game_state['coin_locs']
    players = self.game_state['others']
    bombs = self.game_state['bombs']
    me = self.game_state['self']
    for x, y in coins:
        ind = x_y_to_index(x, y, s.cols, s.rows) - 1
        state[ind] = 3

    for x in range(arena.shape[0]):
        if x != 0 and x != arena.shape[0] - 1:
            for y in range(arena.shape[1]):
                if y != 0 and y != arena.shape[1] - 1 and (x + 1) * (y + 1) % 2 != 1:
                    ind = x_y_to_index(x, y, s.cols, s.rows) - 1
                    coin = state[ind] == 3
                    if not coin:
                        state[ind] = arena[x, y]  # either crates or empty space
                    if explosions[x, y] != 0:
                        state[ind] = -1 * 3 ** int(coin) * 2 ** explosions[x, y]

    startplayers = x_y_to_index(15, 15, s.cols, s.rows)  # player blocks start here
    if me not in players:
        players.insert(0, me)
    bomb_ind = 0
    for player_ind, player in enumerate(players):  # keep track of player locations and bombs
        state[startplayers + player_block * player_ind] = x_y_to_index(player[0], player[1], s.cols, s.rows)
        if player[3] == 0: # if player has no bombs
            player_bomb = bombs[bomb_ind]  # count through bombs and assign a dropped bomb to each player
            # who is not holding a bomb
            state[startplayers + player_block * player_ind + 2] = x_y_to_index(player_bomb[0], player_bomb[1],
                                                                                    s.cols, s.rows)
            state[startplayers + player_block * player_ind + 3] = player_bomb[2]  # bomb timer
            bomb_ind += 1

    return state.astype(int)
