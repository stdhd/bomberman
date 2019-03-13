
import numpy as np
from settings import s


from agent_code.marathon.indices import *


"""
Available Features: 

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
"""


class ObservationObject:
    """
    class to keep track of constants such as window size, number of features, etc..
    """

    def __init__(self, radius, FEATURE_LIST:list):
        """
        Initialize Observation Object using window radius and list of features to create.
        :param radius: Radius in observation window (radius = 1 => 3x3 window)
        :param FEATURE_LIST: list of features by name
        """
        sorted(FEATURE_LIST, key=str.lower)
        self.features = FEATURE_LIST
        self.radius = radius

        self.window_size = (2 * radius + 1)
        self.NUM_FEATURES = len(self.features)
        self.obs_length = self.window_size ** 2 + self.NUM_FEATURES

        self.state, self.board, self.player_locs, self.coins, self.player_distance_matrix = None, None, None, None, None

        self.bomb_locs, self.bomb_timers = None, None

        self.player = None  # helper variable for creating features (not observation window)

    def set_state(self, state):

        """
        Set the state for the observation object and initialize useful feature values.
        :param state:
        :return:
        """

        self.state = state

        self.initialize_feature_helpers()

    def create_observation(self, AGENTS):
        """
        From state, view radius and list of players, return a list of observations.
        :param state: Game state
        :param radius: View window radius
        :param AGENTS: List of player indices (0 to 3 inclusive)
        :return: Array of observations
        """

        radius, board, player_locs, bomb_locs, bomb_timers, window_size = self.radius, self.board, self.player_locs, \
                                                             self.bomb_locs, self.bomb_timers, self.window_size

        observations = np.zeros((AGENTS.shape[0], self.obs_length))

        features = self._get_features(AGENTS)  # find features for all agents

        for count, player_index in enumerate(AGENTS):  # construct the window for all agents
            window = np.zeros((window_size, window_size))

            player_x, player_y = index_to_x_y(self.player_locs[int(player_index)])

            lower_x = player_x - radius

            lower_y = player_y - radius

            for i in np.arange(window_size):
                for j in np.arange(window_size):
                    try:
                        window[i, j] = self.board[x_y_to_index(lower_x + i, lower_y + j) - 1]

                    except Exception as e:  # wall squares throw exception
                        window[i, j] = -1

            for ind, bomb_loc in enumerate(bomb_locs):  # bombs have precedence over explosions
                if bomb_loc > 0:
                    self.set_window(window, bomb_loc, player_x, player_y, radius, 2 ** bomb_timers[ind])

            for player_loc in player_locs:
                if player_loc > 0:
                    location_value = self.get_window(window, *index_to_x_y(player_loc), radius, player_x, player_y)

                    if location_value > 0:  # if player is on a bomb, multiply bomb timer and player value
                        self.set_window(window, player_loc, player_x, player_y, radius, location_value * 5)

                    else:  # else set field to player value
                        self.set_window(window, player_loc, player_x, player_y, radius, 5)

            observations[count] = np.concatenate((window.flatten(), features[count]))  # concatenate window and features

        return observations

    def is_in_window(self, obj_index, origin_x, origin_y, radius):
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

    def get_window(self, window, board_x, board_y, window_radius, window_origin_x, window_origin_y):
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

        if not self.is_in_window(x_y_to_index(board_x, board_y), window_origin_x, window_origin_y, window_radius):
            raise ValueError("Board coordinates not in window. ")

        return window[board_x - (window_origin_x - window_radius), board_y - (window_origin_y - window_radius)]

    def set_window(self, window, board_index, window_origin_x, window_origin_y, window_radius, val):
        if not self.is_in_window(board_index, window_origin_x, window_origin_y, window_radius):
            return

        board_x, board_y = index_to_x_y(board_index)

        window[board_x - (window_origin_x - window_radius), board_y - (window_origin_y - window_radius)] = val


    def initialize_feature_helpers(self):

        if self.state is None:

            raise AttributeError("State not set (call set_state)")

        state = self.state

        board_end = state.shape[0] - (1 + 4 * 21)

        self.board = state[0: board_end]

        player_blocks = state[board_end:]

        self.player_locs = np.array([player_blocks[i * 21] for i in range(4)])  # player locations

        self.coins = np.arange(self.board.shape[0] + 1)[1:][np.where(self.board == 3)]  # list of coin indices

        self.bomb_locs = np.array([player_blocks[i * 21 + 2] for i in range(4)])  # bomb locations

        self.bomb_timers = np.array([player_blocks[i * 21 + 3] for i in range(4)])  # bomb timers

        # manhattan dist. to coins
        self.player_distance_matrix = np.zeros((4, 4))

        for p1 in np.arange(self.player_distance_matrix.shape[0]):
            for p2 in np.arange(start=p1 + 1, stop=self.player_distance_matrix.shape[1]):
                if self.player_locs[p1] == 0 or self.player_locs[p2] == 0:
                    continue  # skip dead players

                self.player_distance_matrix[p1, p2] = np.linalg.norm(np.array([*index_to_x_y(self.player_locs[p1])])
                                                                - np.array([*index_to_x_y(self.player_locs[p2])]), ord=1)

    def _get_features(self, AGENTS):

        """
        Internal function to return features as list.
        :param AGENTS: List of player indices (0 to 3) for which to generate observations.
        :return:
        """

        return_features = np.zeros((AGENTS.shape[0], len(self.features)))

        for count, agent_index in enumerate(AGENTS):
            self.player = _Player(self, agent_index)
            for feature_index, feature in enumerate(self.features):
                method_to_call = getattr(self, feature)
                val = method_to_call()
                return_features[count][feature_index] = val

        return return_features

    def _is_holding_bomb(self, player_index):
        """
        Internal function, do not call.

        Return 1 if player with player_index is currently holding a bomb, else 0
        :param player_index:
        :return:
        """

        begin = self.state.shape[0] - (1 + (4 - player_index) * 21)
        end = self.state.shape[0] - (1 + (4 - player_index - 1) * 21)
        player = self.state[begin: end]
        return 1 if player[2] == 0 else 0

    def _get_path_dir(self, start, end):
        """

        Internal function, do not call.

        Given start and end board indices, get the path direction from start to end.

        Assuming directions are ambiguous (for instance: could go up or left), randomly return a direction.

        In case of start==end or other exception, return 0.

        See conventions/settings.py for path directions.

        :param start:
        :param end:
        :return:
        """

        dirs = {'left': 1, 'right': 2, 'up': 3, 'down': 4, 'except': 0}

        start_x, start_y = index_to_x_y(start)

        end_x, end_y = index_to_x_y(end)

        choose = []

        if start_x > end_x:
            choose.append('left')

        elif end_x > start_x:
            choose.append('right')

        if start_y > end_y:
            choose.append('down')

        elif end_y > start_y:
            choose.append('up')

        if not choose:
            choose.append('except')

        return dirs[np.random.choice(np.array(choose))]

    def me_has_bomb(self):
        """
        Does THIS player currently hold a bomb?
        :param player_index: Index of "me" (player for whom observation is generated)
        :return: 1 if holding bomb, 0 otherwise
        """

        return self._is_holding_bomb(self.player.player_index)

    def closest_coin_dist(self):
        """
        Shortest distance from a certain player to a coin
        :param player_index:
        :return:
        """
        player = self.player
        if player.coin_dists is None:
            player.coin_dists = np.array([np.linalg.norm(player.me_loc - np.array([*index_to_x_y(coin)]), ord=1)
                                          for coin in self.coins])
            player.closest_coin = np.argmin(player.coin_dists)
        # manhattan dist. to coins
        return np.min(player.coin_dists)

    def closest_coin_dir(self):
        """
        Direction to player's nearest coin.
        (See settings.py for encoding 1 - 4)
        :return:
        """

        player = self.player

        if player.coin_dists is None:
            player.coin_dists = np.array([np.linalg.norm(player.me_loc - np.array([*index_to_x_y(coin)]), ord=1)
                                          for coin in self.coins])
            player.closest_coin = np.argmin(player.coin_dists)

        return self._get_path_dir(self.player_locs[player.player_index], self.coins[player.closest_coin])

    def closest_foe_dist(self):
        """
        Distance to player's nearest foe
        :return:
        """

        player = self.player

        player.foe_dists = np.array([self.player_distance_matrix[min(player.player_index, foe_index),
                                                                 max(foe_index, player.player_index)]
                              for foe_index in range(4) if foe_index != player.player_index])
        # manhattan dist. to foes
        player.closest_foe = int(np.argmin(player.foe_dists))  # player index of closest foe

        player.closest_foe += 1 if player.closest_foe >= player.player_index else 0  # me not in foe_dists

        return np.min(player.foe_dists)

    def closest_foe_dir(self):

        """
        Direction to player's nearest foe.
        :return:
        """

        player = self.player

        return self._get_path_dir(self.player_locs[player.player_index], self.player_locs[player.closest_foe])

    def closest_foe_has_bomb(self):
        """
        1 if the closest foe is holding a bomb, 0 otherwise
        :return:
        """
        return self._is_holding_bomb(self.player.closest_foe)

    def nearest_foe_to_closest_coin(self):
        """
        Minimum distance of a foe from MY closest coin
        :return:
        """

        closest_coin_coords = np.array([*index_to_x_y(self.coins[self.player.closest_coin])])

        return np.min(np.array([np.linalg.norm( closest_coin_coords - np.array([*index_to_x_y(foe_loc)]))
                                for foe_loc in self.player.foes]))

    def smallest_enemy_dist(self):
        """
        Smallest distance between two living enemies.
        :return:
        """

        enemies = np.delete(self.player_distance_matrix, self.player.player_index)

        enemies = np.delete(enemies.T, self.player.player_index)

        return np.min(enemies[np.where(enemies != 0)])  # smallest distance between two living enemies

    def dist_to_center(self):
        """
        Manhattan distance to center of map.
        :return:
        """
        center_map = np.array([s.rows//2, s.cols//2])

        return np.linalg.norm(center_map - self.player.me_loc, ord=1)


    def remaining_enemies(self):
        """
        Number of remaining enemies.
        :return:
        """

        return self.player_locs[np.where(self.player_locs != 0)].shape[0]  # count living enemies

    def remaining_crates(self):
        return self.board[np.where(self.board == 1)].shape[0]  # count remaining crates

    def remaining_coins(self):
        return self.coins.shape[0]

    def get_observation_size(self):
        return self.obs_length


class _Player:
    """
    Helper class to store useful attributes for a certain player.
    """

    def __init__(player_self, observation_self, player_index):
        player_self.player_index = player_index
        player_self.me_loc = np.array([*index_to_x_y(observation_self.player_locs[int(player_index)])])
        player_self.coin_dists, player_self.closest_coin = None, None  # distances of all coins, index of closest coin
        player_self.foes = [foe_loc for ind, foe_loc in enumerate(observation_self.player_locs) if ind != player_index]
        player_self.foe_dists, player_self.closest_foe = None, None

