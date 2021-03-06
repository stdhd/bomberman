from settings import s
import numpy as np

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


x_y_to_index_data = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 16, 24, 39, 47, 62, 70, 85, 93, 108, 116, 131, 139, 154, 162, -1],
    [-1, 2, -1, 25, -1, 48, -1, 71, -1, 94, -1, 117, -1, 140, -1, 163, -1],
    [-1, 3, 17, 26, 40, 49, 63, 72, 86, 95, 109, 118, 132, 141, 155, 164, -1],
    [-1, 4, -1, 27, -1, 50, -1, 73, -1, 96, -1, 119, -1, 142, -1, 165, -1],
    [-1, 5, 18, 28, 41, 51, 64, 74, 87, 97, 110, 120, 133, 143, 156, 166, -1],
    [-1, 6, -1, 29, -1, 52, -1, 75, -1, 98, -1, 121, -1, 144, -1, 167, -1],
    [-1, 7, 19, 30, 42, 53, 65, 76, 88, 99, 111, 122, 134, 145, 157, 168, -1],
    [-1, 8, -1, 31, -1, 54, -1, 77, -1, 100, -1, 123, -1, 146, -1, 169, -1],
    [-1, 9, 20, 32, 43, 55, 66, 78, 89, 101, 112, 124, 135, 147, 158, 170, -1],
    [-1, 10, -1, 33, -1, 56, -1, 79, -1, 102, -1, 125, -1, 148, -1, 171, -1],
    [-1, 11, 21, 34, 44, 57, 67, 80, 90, 103, 113, 126, 136, 149, 159, 172, -1],
    [-1, 12, -1, 35, -1, 58, -1, 81, -1, 104, -1, 127, -1, 150, -1, 173, -1],
    [-1, 13, 22, 36, 45, 59, 68, 82, 91, 105, 114, 128, 137, 151, 160, 174, -1],
    [-1, 14, -1, 37, -1, 60, -1, 83, -1, 106, -1, 129, -1, 152, -1, 175, -1],
    [-1, 15, 23, 38, 46, 61, 69, 84, 92, 107, 115, 130, 138, 153, 161, 176, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
])

index_to_x_y_data = np.array([
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 1),
    (9, 1),
    (10, 1),
    (11, 1),
    (12, 1),
    (13, 1),
    (14, 1),
    (15, 1),
    (1, 2),
    (3, 2),
    (5, 2),
    (7, 2),
    (9, 2),
    (11, 2),
    (13, 2),
    (15, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (6, 3),
    (7, 3),
    (8, 3),
    (9, 3),
    (10, 3),
    (11, 3),
    (12, 3),
    (13, 3),
    (14, 3),
    (15, 3),
    (1, 4),
    (3, 4),
    (5, 4),
    (7, 4),
    (9, 4),
    (11, 4),
    (13, 4),
    (15, 4),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5),
    (5, 5),
    (6, 5),
    (7, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (11, 5),
    (12, 5),
    (13, 5),
    (14, 5),
    (15, 5),
    (1, 6),
    (3, 6),
    (5, 6),
    (7, 6),
    (9, 6),
    (11, 6),
    (13, 6),
    (15, 6),
    (1, 7),
    (2, 7),
    (3, 7),
    (4, 7),
    (5, 7),
    (6, 7),
    (7, 7),
    (8, 7),
    (9, 7),
    (10, 7),
    (11, 7),
    (12, 7),
    (13, 7),
    (14, 7),
    (15, 7),
    (1, 8),
    (3, 8),
    (5, 8),
    (7, 8),
    (9, 8),
    (11, 8),
    (13, 8),
    (15, 8),
    (1, 9),
    (2, 9),
    (3, 9),
    (4, 9),
    (5, 9),
    (6, 9),
    (7, 9),
    (8, 9),
    (9, 9),
    (10, 9),
    (11, 9),
    (12, 9),
    (13, 9),
    (14, 9),
    (15, 9),
    (1, 10),
    (3, 10),
    (5, 10),
    (7, 10),
    (9, 10),
    (11, 10),
    (13, 10),
    (15, 10),
    (1, 11),
    (2, 11),
    (3, 11),
    (4, 11),
    (5, 11),
    (6, 11),
    (7, 11),
    (8, 11),
    (9, 11),
    (10, 11),
    (11, 11),
    (12, 11),
    (13, 11),
    (14, 11),
    (15, 11),
    (1, 12),
    (3, 12),
    (5, 12),
    (7, 12),
    (9, 12),
    (11, 12),
    (13, 12),
    (15, 12),
    (1, 13),
    (2, 13),
    (3, 13),
    (4, 13),
    (5, 13),
    (6, 13),
    (7, 13),
    (8, 13),
    (9, 13),
    (10, 13),
    (11, 13),
    (12, 13),
    (13, 13),
    (14, 13),
    (15, 13),
    (1, 14),
    (3, 14),
    (5, 14),
    (7, 14),
    (9, 14),
    (11, 14),
    (13, 14),
    (15, 14),
    (1, 15),
    (2, 15),
    (3, 15),
    (4, 15),
    (5, 15),
    (6, 15),
    (7, 15),
    (8, 15),
    (9, 15),
    (10, 15),
    (11, 15),
    (12, 15),
    (13, 15),
    (14, 15),
    (15, 15)])


def x_y_to_index(x, y, ncols=s.cols, nrows=s.rows):
    """
    Return the index of a free grid from x, y coordinates. Indices start at 1 !

    Indexing starts at x, y = 0, 0 and increases row by row. Higher y value => Higher index

    Raises ValueError for wall coordinates!
    :param x: x coordinate
    :param y: y coordinate
    :return: Index of square x, y coords point to
    """

    x = int(x)
    y = int(y)

    if x >= ncols - 1 or x <= 0 or y >= nrows - 1 or y <= 0:
        raise ValueError("Coordinates outside of game grid")
    if (x + 1) * (y + 1) % 2 == 1:
        raise ValueError("Received wall coordinates!")

    return x_y_to_index_data[x, y]


def index_to_x_y(ind, ncols=s.cols, nrows=s.rows):
    """
    Convert a given coordinate index into its x, y representation. Indices start at 1 !
    :param ind: Index of coordinate to represent as x, y
    :param ncols: Number of
    :param nrows:
    :return: x, y coordinates
    """

    ind = int(ind)

    if ind == 0:
        raise ValueError("Got zero index (dead player)")

    if ind > 176 or ind < 1:
        raise ValueError("Index out of range. (max 176, min 1, got", ind, ")")

    return index_to_x_y_data[ind - 1]