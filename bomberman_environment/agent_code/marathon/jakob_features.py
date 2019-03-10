
import numpy as np

from settings import s

from agent_code.marathon.indices import *


def jakob_features(state, player_index=0, ncols=s.cols, nrows=s.rows):
    """
    Given a state vector, return a list of features for a player as an array. (See Google Drive: features)

    :param state:
    :param player_index: Which player to generate the features for.
    :return: features in array form
    """

    board_end = state.shape[0] - (1 + 4 * 21)

    board = state[0: board_end]

    player_blocks = state[board_end:]

    player_locs = np.array([player_blocks[i*21] for i in range(4)])  # player locations

    me_loc = np.array([*index_to_x_y(player_locs[player_index])])

    me_has_bomb = is_holding_bomb(state, player_index)  # indicate whether player is holding a bomb

    coins = np.arange(board.shape[0]+1)[1:][np.where(board == 3)]  # list of coin indices

    coin_dists = np.array([np.linalg.norm(me_loc - np.array([*index_to_x_y(coin)]), ord=1) for coin in coins])
    # manhattan dist. to coins

    closest_coin = np.argmin(coin_dists)

    closest_coin_dist = coin_dists[closest_coin]

    closest_coin_dir = get_path_dir(player_locs[player_index], coins[closest_coin])

    player_distance_matrix = np.zeros((4, 4))

    for p1 in np.arange(player_distance_matrix.shape[0]):
        for p2 in np.arange(start=p1 + 1, stop=player_distance_matrix.shape[1]):
            if player_locs[p1] == 0 or player_locs[p2] == 0:
                continue # skip dead players

            player_distance_matrix[p1, p2] = np.linalg.norm(np.array([*index_to_x_y(player_locs[p1])])
                                                            - np.array([*index_to_x_y(player_locs[p2])]), ord=1)

    foes = [foe_loc for ind, foe_loc in enumerate(player_locs) if ind != player_index]

    foe_dists = np.array([player_distance_matrix[min(player_index, foe_index), max(foe_index, player_index)]
                          for foe_index in range(4) if foe_index != player_index])
    # manhattan dist. to foes

    closest_foe = int(np.argmin(foe_dists))  # player index of closest foe

    closest_foe += 1 if closest_foe >= player_index else 0  # me not in foe_dists

    closest_foe_dist = np.min(foe_dists)

    closest_foe_dir = get_path_dir(player_locs[player_index], player_locs[closest_foe])

    closest_foe_has_bomb = is_holding_bomb(state, closest_foe)

    closest_coin_coords = np.array([*index_to_x_y(coins[closest_coin])])

    nearest_foe_to_closest_coin = np.min(np.array([np.linalg.norm( closest_coin_coords -
                                                    np.array([*index_to_x_y(foe_loc)])) for foe_loc in foes]))
    #  minimum distance of a foe from MY closest coin

    enemies = np.delete(player_distance_matrix, player_index)

    enemies = np.delete(enemies.T, player_index)

    smallest_enemy_dist = np.min(enemies[np.where(enemies != 0)])  # smallest distance between two living enemies

    center_map = np.array([s.rows//2, s.cols//2])

    dist_to_center = np.linalg.norm(center_map - me_loc)

    remaining_enemies = player_locs[np.where(player_locs != 0)].shape[0]  # count living enemies

    remaining_crates = board[np.where(board == 1)].shape[0]  # count remaining crates

    remaining_coins = coins.shape[0]

    return np.array([
        me_has_bomb,
        closest_coin_dist,
        closest_coin_dir,
        closest_foe_dist,
        closest_foe_dir,
        closest_foe_has_bomb,
        nearest_foe_to_closest_coin,
        smallest_enemy_dist,
        dist_to_center,
        remaining_enemies,
        remaining_crates,
        remaining_coins
    ])


def is_holding_bomb(state, player_index):
    """
    Return 1 if player with player_index is currently holding a bomb, else 0
    :param state:
    :param player_index:
    :return:
    """

    begin = state.shape[0] - (1 + (4 - player_index) * 21)
    end = state.shape[0] - (1 + (4 - player_index - 1) * 21)
    player = state[begin: end]

    return 1 if player[2] == 0 else 0


def get_path_dir(start, end):

    """
    Given start and end board indices, get the path direction from start to end.

    Assuming directions are ambiguous (for instance: could go up or left), randomly return a direction.

    See conventions for path directions.

    :param start:
    :param end:
    :return:
    """

    dirs = {'left': 1, 'right': 2, 'up': 3, 'down': 4, 'except': 0}

    start_x, start_y = index_to_x_y(start)

    end_x, end_y = index_to_x_y(end)

    choose = []

    if start_x > end_x: choose.append('left')

    elif end_x > start_x: choose.append('right')

    if start_y > end_y: choose.append('down')

    elif end_y > start_y: choose.append('up')

    if not choose:
        choose.append('except')

    return dirs[np.random.choice(np.array(choose))]










