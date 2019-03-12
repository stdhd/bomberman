
import numpy as np
from agent_code.marathon.indices import *
from agent_code.marathon.jakob_features import jakob_features


def is_in_window(obj_index, origin_x, origin_y, radius):
    """
    Given the origin of the view window and its radius, determine whether an object lies in the view window.
    :param origin_x: Window origin (x coord)
    :param origin_y: Window origin (y coord)
    :param radius: Radius of window
    :return: True iff in window
    """

    obj_x, obj_y = index_to_x_y(obj_index)

    if abs(obj_x - origin_x) > radius or abs(obj_y - origin_y) > radius:
        return False

    return True

def get_window(window, board_x, board_y, window_radius, window_origin_x, window_origin_y):
    """
    Get the value of the window field corresponding to board field x, y.
    :param window
    :param board_x:
    :param board_y:
    :param window_radius:
    :param window_origin_x:
    :param window_origin_y:
    :return:
    """

    if not is_in_window(x_y_to_index(board_x, board_y)):
        raise ValueError("Board coordinates not in window. ")

    return window[board_x - (window_origin_x - window_radius), board_y - (window_origin_y - window_radius)]

def set_window(window, board_index, window_origin_x, window_origin_y, window_radius, val):

    if not is_in_window(board_index, window_origin_x, window_origin_y, window_radius):
        return

    board_x, board_y = index_to_x_y(board_index)

    window[board_x - (window_origin_x - window_radius), board_y - (window_origin_y - window_radius)] = val


def create_observation(state, radius, player_indices):
    """
    From state, view radius and list of players, return a list of observations.
    :param state: Game state
    :param radius: View window radius
    :param player_indices: List of player indices (0 to 3 inclusive)
    :return: Array of observations
    """

    board_end = state.shape[0] - (1 + 4 * 21)

    board = state[0: board_end]

    player_blocks = state[board_end:]

    player_locs = np.array([player_blocks[i*21] for i in range(4)])  # player locations

    bomb_locs = np.array([player_blocks[i*21 + 2] for i in range(4)])  # bomb locations

    bomb_timers = np.array([player_blocks[i*21 + 3] for i in range(4)])  # bomb timers

    window_size = (2*radius + 1)

    NUM_FEATURES = 12

    observations = np.zeros((player_indices.shape[0], window_size**2 + NUM_FEATURES))

    for count, player_index in enumerate(player_indices):

        window = np.zeros((window_size, window_size))

        player_x, player_y = index_to_x_y(player_index)

        lower_x = player_x - radius

        lower_y = player_y - radius

        for i in np.arange(window_size):
            for j in np.arange(window_size):

                try:

                    window[i, j] = board[x_y_to_index(lower_x + i, lower_y + j) - 1]

                except Exception as e:  # wall squares throw exception

                    window[i, j] = -1

        for ind, bomb_loc in enumerate(bomb_locs):  # bombs have precedence over explosions

            set_window(window, bomb_loc, player_x, player_y, radius, 2**bomb_timers[ind])

        for player_loc in player_locs:

            location_value = get_window(window, *index_to_x_y(player_loc), radius, player_x, player_y)

            if location_value > 0: # if player is on a bomb, multiply bomb timer and player value

                set_window(window, player_loc, player_x, player_y, radius, location_value*5)

            else: # else set field to player value

                set_window(window, player_loc, player_x, player_y, radius, 5)

        observations[count] = np.concatenate((window.flatten(), jakob_features(state, player_index)))




