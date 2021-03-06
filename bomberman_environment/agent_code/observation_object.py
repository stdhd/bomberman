from state_functions.indices import *
from state_functions.rewards import get_reward
from random import shuffle
import numpy as np
import settings



"""
Available Features: 

        me_has_bomb,
        closest_coin_dist,
        d_closest_coin_dir,
        d_closest_crate_dir,
        closest_foe_dist,
        d_closest_foe_dir,
        closest_foe_has_bomb,
        nearest_foe_to_closest_coin,
        smallest_enemy_dist,
        dist_to_center,
        remaining_enemies,
        remaining_crates,
        remaining_coins,
        closest_coin_old,
        d_closest_safe_field_dir
        d4_is_safe_to_move_a_l
        d4_is_safe_to_move_b_r
        d4_is_safe_to_move_c_u
        d4_is_safe_to_move_d_d
        d_closest_enemy_dir
        d_best_bomb_dropping_dir
        enemy_in_bomb_area
"""


class ObservationObject:
    """
    class to keep track of constants such as window size, number of features, etc..
    """

    def __init__(self, radius, FEATURE_LIST:list, logger):
        """
        Initialize Observation Object using window radius and list of features to create.
        :param radius: Radius in observation window (radius = 1 => 3x3 window)
        :param FEATURE_LIST: list of features by name
        """
        self.features = FEATURE_LIST
        self.features.sort()

        self.radius = radius
        self.logger = logger

        self.window_size = (2 * radius + 1)
        self.NUM_FEATURES = len(self.features)
        if radius == -1:
            self.obs_length = self.NUM_FEATURES
        else:
            self.obs_length = self.window_size ** 2 + self.NUM_FEATURES

        self.state, self.board, self.player_locs, self.coin_locs, self.player_distance_matrix = None, None, None, None, None

        self.arena = None

        self.bomb_locs, self.bomb_timers, self.dead_players = None, None, None

        self.player = None  # helper variable for creating features (not observation window)

        self.dead_players, self.just_died, self.living_players = np.array([]), np.array([]), None


        self.name_dict = {
        "me_has_bomb": "mhb",
        "closest_coin_dist": "ccdist",
        "d_closest_coin_dir": "ccdir",
        "d_closest_crate_dir": "ccrdir",
        "closest_foe_dist": "cfdist",
        "d_closest_foe_dir": "cfdir",
        "closest_foe_has_bomb": "cfhb",
        "dead_end_detect": "ded",
        "nearest_foe_to_closest_coin": "nftcc",
        "dist_to_center": "dtc",
        "smallest_enemy_dist": "smd",
        "remaining_enemies": "re",
        "remaining_crates": "rcrates",
        "remaining_coins": "rcoins",
        "d_closest_safe_field_dir": "csfdir",
        "closest_coin_old": "cco",
        "d4_is_safe_to_move_a_l" : "ismal",
        "d4_is_safe_to_move_b_r": "ismbr",
        "d4_is_safe_to_move_c_u": "ismcu",
        "d4_is_safe_to_move_d_d": "ismdd",
        "d_closest_enemy_dir": "ced",
        "d_best_bomb_dropping_dir":"bbdd",
        "enemy_in_bomb_area":"eiba"
        }

    def set_state(self, state):

        """
        Set the state for the observation object and initialize useful feature values.
        :param state:
        :return:
        """

        self.state = state

        _old_died = self.dead_players.copy()

        self._initialize_feature_helpers()

        self.just_died = np.array([player for player in self.dead_players if player not in _old_died]).astype(int)


    def _initialize_feature_helpers(self):
        if self.state is None:
            raise AttributeError("State not set (call set_state)")

        state = self.state
        board_end = state.shape[0] - (1 + 4 * 21)
        self.board = state[0: board_end]
        player_blocks = state[board_end:]
        self.player_locs = np.array([player_blocks[i * 21] for i in range(4)])  # player locations
        self.coin_locs = np.where(self.board == 3)[0] + 1  # list of coin indices
        self.bomb_locs = np.array([player_blocks[i * 21 + 2] for i in range(4)])  # bomb locations
        self.bomb_timers = np.array([player_blocks[i * 21 + 3] for i in range(4)])  # bomb timers
        # killed_booleans = np.array([player_blocks[i * 21 + 18] for i in range(4)])  # note "got killed" boolean
        self.dead_players = np.where(self.player_locs == 0)[0]
        self.living_players = np.where(self.player_locs != 0)[0]
        self.events = np.array([player_blocks[i*21 + 4: (i + 1)*21] for i in range(4)])
        # get (4 x 17) matrix of events for this step
        self.arena = self._make_window(8, 8, 8)
        self.danger_map = self._get_threat_map()
        self.number_of_remaining_crates = np.sum(self.arena == 1)
        # if self.logger: 
            # self.logger.info(f'DANGER MAP: {1*self.danger_map}')
            # self.logger.info(f'ARENA: {self.arena}')


    def reset_killed_players(self):
        """
        Call this when starting a new game to refresh information about "just died" players
        :return:
        """
        self.dead_players, self.just_died = np.array([]), np.array([])

    def create_observation(self, AGENTS:np.array):
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
            player_x, player_y = index_to_x_y(self.player_locs[int(player_index)])
            if self.radius == -1:
                window = np.array([])
            else:
                window = self._make_window(radius, player_x, player_y)
            observations[count] = np.concatenate((window.flatten(), features[count]))  # concatenate window and features
        return observations

    def get_feature_index(self, feature_name):
        try:
            if self.radius == -1:
                return self.features.index(feature_name)
            else:
                return self.features.index(feature_name) + 2 * self.radius + 1
        except Exception as e:
            return None

    def _make_window(self, radius_custom, center_x, center_y):
        """
        Creates a window centered on given coordinates.
        :param radius:
        :param center_x:
        :param center_y:
        :return:
        """
        window_size_custom = 2*radius_custom + 1
        window = np.zeros((window_size_custom, window_size_custom))
        lower_x = center_x - radius_custom
        lower_y = center_y - radius_custom

        for i in np.arange(window_size_custom):
            for j in np.arange(window_size_custom):
                try:
                    temp = self.board[x_y_to_index(lower_x + i, lower_y + j) - 1]  # note coins, crates, explosions, and free

                    if temp == -2:  # FIXME ignore explosions with timer <= 1
                        continue
                    elif temp == -6:  # FIXME
                        temp = 3
                    window[i, j] = temp
                except ValueError as e:  # wall squares throw exception
                    window[i, j] = -1

        for player_loc in self.player_locs:
            if player_loc > 0:
                try:
                    self.set_window(window, player_loc, center_x, center_y, radius_custom, 5)
                except ValueError:
                    continue

        for ind, bomb_loc in enumerate(self.bomb_locs):  # bombs have precedence over explosions and players
            if bomb_loc > 0:
                if self.bomb_timers[ind] <= 1:
                    self.set_window(window, bomb_loc, center_x, center_y, radius_custom, 2)  # FIXME revert change
                    # FIXME bombs with 1 or 0 now indicate 2
                else:
                    self.set_window(window, bomb_loc, center_x, center_y, radius_custom, 4)

        return window

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
    
    def _look_for_targets(self, free_space, start, targets, logger):
        """Find direction of closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards closest target or towards tile closest to any target.
        """
        if len(targets) == 0: return None
        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        if logger: logger.debug(f'Suitable target found at {best}')
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]

    
    def _look_for_targets_safe_field(self, free_space, start, targets, logger):
        """Find direction of closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True, for free tiles and False, for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards closest target or towards tile closest to any target.
        """
        if len(targets) == 0: return None
        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
        targets_reachable = False
        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                targets_reachable = True
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        if logger: logger.debug(f'Suitable target found at {best}')
        # Determine the first step towards the best found target tile
        current = best
        if (not targets_reachable):
            return 10
        while True:
            if current == start: print("Komisch", current)
            if parent_dict[current] == start: return current
            current = parent_dict[current]

    def _look_for_targets_weighted(self, free_space, start, targets, targets_weights, logger):
        """Find direction of closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            targets_weights: array holding the targets' weights. The higher the weight the less important it is.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards closest target or towards tile closest to any target.
        """
        if len(targets) == 0: return None
        # Check if it is safe to place a bomb at the current position. If not delete current position from targets.
        bombs = np.copy(self.bomb_locs)
        bombs[0] = x_y_to_index(start[0], start[1])
        np.array([99,0,0,0])
        safe_field_dir = self.d_closest_safe_field_dir(bombs)
        # if self.logger: self.logger.info(f'safe_field_dir wighted {safe_field_dir}')
        if safe_field_dir == 6:

            if self.logger: self.logger.info(f'Deleted: {targets[np.where((targets == np.array([start[0], start[1]])).all(axis=1))[0]]}')

            # self.logger.info(f'Deleted: {targets[np.where((targets == np.array([start[0], start[1]])).all(axis=1))[0]]}')

            current_pos_ind = np.where((targets == np.array([start[0], start[1]])).all(axis=1))[0]
            if current_pos_ind.shape[0] != 0:
                targets = np.delete(targets, current_pos_ind[0], axis=0)
        # if self.logger: self.logger.info(f'Targetts {targets}')
        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = (np.sum(np.abs(np.subtract(targets, start)), axis=1)).min()
        best_candidates_with_bomb_value = {}
        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = (np.sum(np.abs(np.subtract(targets, current)), axis=1)).min()  # fixme:  * targets_weights
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Find all possible crate bombing positions
                best = current
                cand_ind = np.where((targets == best).all(axis=1))[0][0]
                best_candidates_with_bomb_value[best] = targets_weights[cand_ind]
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        if logger: logger.debug(f'Suitable target found at {best}')
        max_value = 0
        # Choose best crate bombing position by selecting max of bomb_value/(dist+1)
        if len(best_candidates_with_bomb_value) == 0: return None
        bomb_value_keys =  list(best_candidates_with_bomb_value.keys())      # Python 3; use keys = d.keys() in Python 2
        shuffle(bomb_value_keys)
        best_candidates_with_bomb_value_list = [(key, best_candidates_with_bomb_value[key]) for key in bomb_value_keys]
        for cand in best_candidates_with_bomb_value_list:
            cand_value = cand[1] / (dist_so_far[cand[0]] + 1) # Avoid dividing by 0
            if cand_value > max_value:
                max_value = cand_value
                current = cand[0]
        # Determine the first step towards the best found target tile
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]

    def _look_for_targets_coins(self, free_space, start, targets, logger):
        """Find direction of closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards closest target or towards tile closest to any target.
        """
        if len(targets) == 0: return None
        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
        targets_reachable = False
        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                targets_reachable = True
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        if logger: logger.debug(f'Suitable target found at {best}')
        # Determine the first step towards the best found target tile
        current = best

        if (not targets_reachable):
            return None

        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]

    def _determine_direction(self, best_step, x, y):
        if best_step == (x-1,y): return 0 # move left
        if best_step == (x+1,y): return 1 # move right
        if best_step == (x,y-1): return 2 # move up
        if best_step == (x,y+1): return 3 # move down
        if best_step == None: return 4 # No targets exist.
        if best_step == (x,y): return 5 # Target can not be reached. Only occurs if current position is right before the obstacle.
        return 6 # Something else is wrong: This case should not occur

    def me_has_bomb(self):
        """
        Does THIS player currently hold a bomb?
        :param player_index: Index of "me" (player for whom observation is generated)
        :return: 1 if holding bomb, 0 otherwise
        """
        return self._is_holding_bomb(self.player.player_index)

    def d_closest_coin_dist(self):
        """
        Shortest distance from a certain player to a coin
        :param player_index:
        :return:
        """
        player = self.player
        if player.coin_dists is None:
            player.coin_dists = np.array([np.linalg.norm(player.me_loc - np.array([*index_to_x_y(coin)]), ord=1)
                                          for coin in self.coin_locs])
            player.closest_coin = np.argmin(player.coin_dists)
        # manhattan dist. to coin_locs
        return np.min(player.coin_dists)


    def d_closest_coin_dir(self):
        """
        Direction to player's nearest coin.
        :return: 0 (left), 1 (right), 2 (up), 3 (down), 4 (no crate), 5 (one field before not reachable coin), 6 (something went wrong)
        """
        coins_ind = np.where(self.arena == 3)
        coins_coords = np.vstack((coins_ind[0], coins_ind[1])).T
        free_space = (self.arena == 0) | (self.arena == 3)
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        best_step = self._look_for_targets_coins(free_space, (x, y), coins_coords, None)

        return self._determine_direction(best_step, x, y)

    def d_closest_enemy_dir(self):
        """
        Direction to player's nearest enemy.
        """

        x, y = self.player.me_loc[0], self.player.me_loc[1]

        if self.player.foes.shape[0] == 0:
            return self._determine_direction(None, x, y)  # FIXME deactivate when no enemies

        arena = np.copy(self.arena)

        # remove myself from arena
        arena[x, y] = 0
        # Switch feature on when there are less than 11 crates on the field
        if self.number_of_remaining_crates < 11:
            # enemy_coords = np.vstack((enemy_ind[0], enemy_ind[1])).T
            # if self.logger: self.logger.info(f'ENEMY: {self.player.foes}')
            free_space = (arena == 0) | (arena == 3)
            enemy_coords = []
            for enemy in self.player.foes:
                enemy_coords.append([*index_to_x_y(enemy)])
            # if self.logger: self.logger.info(f'ENEMY targets: {enemy_coords}')
            best_step = self._look_for_targets(free_space, (x, y), np.array(enemy_coords), None)
            return self._determine_direction(best_step, x, y)
        else:
            return self._determine_direction(None, x, y)

    def d_closest_crate_dir(self):
        """
        Direction to player's nearest crate. 
        :return: 0 (left), 1 (right), 2 (up), 3 (down), 4 (no crate), 5 (one field before crate), 6 (something went wrong)
        """
        crate_ind = np.where(self.arena == 1)
        crate_coords = np.vstack((crate_ind[0], crate_ind[1])).T
        free_space = (self.arena == 0) | (self.arena == 3)
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        best_step = self._look_for_targets(free_space, (x, y), crate_coords, None)
        return self._determine_direction(best_step, x, y)

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

    def enemy_in_bomb_area(self):
        """
        Return 1 if enemy is in 5 field radius and less than 11 crates are on the field
        Return 0 otherwise
        """
        # if self.logger: self.logger.info(f'FsOES: {self.player.foes}')
        if self.player.foes.shape[0] != 0:
            enemy_coords = []
            for enemy in self.player.foes:
                enemy_coords.append([*index_to_x_y(enemy)])
            best_dist = np.sum(np.abs(np.subtract(np.array(enemy_coords), self.player.me_loc)), axis=1).min()
        else:
            return 0
        # if self.logger: self.logger.info(f'ENEMY CORDS: {np.array(enemy_coords)}')
        if best_dist < 6 & self.number_of_remaining_crates < 11:
            return 1
        else:
            return 0

    def d_closest_foe_dir(self):
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
        closest_coin_coords = np.array([*index_to_x_y(self.coin_locs[self.player.closest_coin])])
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
        return self.coin_locs.shape[0]

    def get_observation_size(self):
        """
        Integer of the total size of the observation vector
        :return:
        """
        return self.obs_length

    def get_file_name_string(self):
        """
        String contsining the short names of the current feature configuration
        :return:
        """
        temp = "r" + str(self.radius)
        self.features = sorted(self.features)
        for i,full_name in enumerate(self.features):
                temp = temp + "_" + self.name_dict[full_name]

        return temp

    def get_direction_sensitivity(self):
        """
        Boolean array indicating, if features are direction sensitive
        :return:
        """
        temp = []
        for f in self.features:
            if f.startswith("d_"):
                temp.append(1)
            elif f.startswith("d4_"):
                temp.append(2)
            else:
                temp.append(0)
        return temp

    def closest_coin_old(self):
        """
        Direction to player's nearest coin (outdated).
        :return:
        """
        player = self.player
        return self._get_path_dir(self.player_locs[player.player_index], self.player_locs[player.closest_coin])


    def _get_threat_map(self, arena_bool=None, bomb_locs=None):
        """
        :return: Boolean map: True for Free/Coin, False for Wall/Threatened/Crate
        """
        if arena_bool is None:
            arena_bool = (self.arena == 0) | (self.arena == 3) | (self.arena == 5)
        if bomb_locs is None:
            bomb_locs = self.bomb_locs
        for loc in bomb_locs:
            if loc == 0:
                continue
            tx,ty = index_to_x_y(loc)
            arena_bool[tx, ty] = False
            for i in range(3):
                if self.arena[tx, ty + (i + 1)] == -1:
                    break
                arena_bool[tx, ty + (i + 1)] = False
            for i in range(3):
                if self.arena[tx, ty - (i + 1)] == -1:
                    break
                arena_bool[tx, ty - (i + 1)] = False
            for i in range(3):
                if self.arena[tx + (i + 1), ty] == -1:
                    break
                arena_bool[tx + (i + 1), ty] = False
            for i in range(3):
                if self.arena[tx - (i + 1), ty] == -1:
                    break
                arena_bool[tx - (i + 1), ty] = False
        return arena_bool

    def d_closest_safe_field_dir(self, bomb_locs=None):
        """
        Direction to next safe field. If no field is found try again without considering explosions.
        bomb_locs can be passed in for the case to check wheter a closest_safe_field is available for a special case
        """
        danger_map = np.copy(self.danger_map)
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        
        # TODO: Make functions more reusable so that this ugly piece of code disappears
        if bomb_locs is None:
            bomb_locs = self.bomb_locs
        else: # Case when closest_safe_field_dir is called from _look_for_targets_weighted with virtual bomb
            free_space = (self.arena == 0) | (self.arena == 3)
            danger_map = self._get_threat_map(free_space, bomb_locs)
            
        
        # If there are no bombs on the field the direction should indicate this by turning off this feature (return 4)
        # if self.logger: self.logger.info(f'CHECK BOMBS: {self.bomb_locs, self.bomb_locs.any()}')
        if (not bomb_locs.any()): 
            # if self.logger: self.logger.info(f'NO BOMBS')
            return self._determine_direction(None, x, y)
        # If agent is not on danger zone indicate this by turning off feature (return 4)
        # if self.logger: self.logger.info(f'On Danger Map: {not self.danger_map[x, y]}')
        if danger_map[x, y]:
            # if self.logger: self.logger.info(f'NOT ON DANGER ZONE')
            return self._determine_direction(None, x, y)
        free_space = (self.arena == 0) | (self.arena == 3)
        free_space_ind = np.where(danger_map == True)
        free_space_coords = np.vstack((free_space_ind[0], free_space_ind[1])).T
        best_step = self._look_for_targets_safe_field(free_space, (x, y), free_space_coords, None)
         # If no safe field is reachable search again with ignored explosion fields
        if best_step == 10:
            # Explosion convention: -1 * 3^(coin_is_present) * 2^(explosion timer if explosion timer is greater than 0)
            free_space = (self.arena == 0) | (self.arena == 3) | (self.arena == -2) | (self.arena == -4) | (self.arena == -6) | (self.arena == -12)
            danger_map_without_explosions = np.copy(free_space)
            danger_map_without_explosions = self._get_threat_map(danger_map_without_explosions, bomb_locs)
            # for loc in bomb_locs:
            #     if loc == 0:
            #         continue
            #     tx,ty = index_to_x_y(loc)
            #     danger_map_without_explosions[tx, ty] = False
            #     for i in range(3):
            #         if self.arena[tx, ty + (i + 1)] == -1:
            #             break
            #         danger_map_without_explosions[tx, ty + (i + 1)] = False
            #     for i in range(3):
            #         if self.arena[tx, ty - (i + 1)] == -1:
            #             break
            #         danger_map_without_explosions[tx, ty - (i + 1)] = False
            #     for i in range(3):
            #         if self.arena[tx + (i + 1), ty] == -1:
            #             break
            #         danger_map_without_explosions[tx + (i + 1), ty] = False
            #     for i in range(3):
            #         if self.arena[tx - (i + 1), ty] == -1:
            #             break
            #         danger_map_without_explosions[tx - (i + 1), ty] = False

            free_space_ind = np.where(danger_map_without_explosions == True)
            free_space_coords = np.vstack((free_space_ind[0], free_space_ind[1])).T
            best_step = self._look_for_targets_safe_field(free_space, (x, y), free_space_coords, None)
        # self.logger.info(f'XY_BOMBS: {np.vstack((x_bombs, y_bombs)).T}')
        # self.logger.info(f'Free Space Coords: {free_space_coords}')
        # self.logger.info(f'Self: {x, y}')
        # if self.logger: self.logger.info(f'Best_step: {best_step}')
        return self._determine_direction(best_step, x, y)

    def d4_is_safe_to_move_a_l(self):
        x,y = self.player.me_loc[0], self.player.me_loc[1]
        if self.danger_map[x - 1, y]:
            return 1
        return 0

    def d4_is_safe_to_move_b_r(self):
        x,y = self.player.me_loc[0], self.player.me_loc[1]
        if self.danger_map[x + 1, y]:
            return True
        return False

    def d4_is_safe_to_move_c_u(self):
        x,y = self.player.me_loc[0], self.player.me_loc[1]
        if self.danger_map[x, y - 1]:
            return True
        return False

    def d4_is_safe_to_move_d_d(self):
        x,y = self.player.me_loc[0], self.player.me_loc[1]
        if self.danger_map[x, y + 1]:
            return True
        return False

    def dead_end_detect(self):
        """
        True if a "dead end" (Sackgasse) situation is present (you are standing on a bomb and could entrap yourself
        by walking in a certain direction).

        True if by laying a bomb now, you would entrap yourself. #FIXME implement this

        Agent should learn to follow the direction
        to safe space in this case.
        :return:
        """

        arena = self.arena
        x, y = self.player.me_loc[0], self.player.me_loc[1]

        if int(arena[x, y]) not in [2, 4]:  # activate only when standing on a bomb
            return 0
        
        blocking = (4, 5, -1, 1)  # objects that block movement (no small bomb timer because it disappears shortly)
        
        navigable_blast_coords = [(x,y)]

        for i in range(1, 3+1):
            if arena[x+i,y] in blocking: break
            navigable_blast_coords.append((x+i,y))
        for i in range(1, 3+1):
            if arena[x-i,y] in blocking: break
            navigable_blast_coords.append((x-i,y))
        for i in range(1, 3+1):
            if arena[x,y+i] in blocking: break
            navigable_blast_coords.append((x,y+i))
        for i in range(1, 3+1):
            if arena[x,y-i] in blocking: break
            navigable_blast_coords.append((x,y-i))

        #
        # nearby = [(x_, y_) for x_ in range(x - 4, x + 5) for y_ in range(y - 4, y + 5) if
        #           (x_ == x) or (y_ == y )] Problem with this is that it also looks at potentially inaccessible fields
        nearby_dead_ends = [pos for pos in navigable_blast_coords if self._is_lethal_dead_end(*pos, x, y )]
        # counts dead ends within killing radius from this point

        return int(len(nearby_dead_ends) > 0)

    def _is_lethal_dead_end(self, x, y, bomb_x, bomb_y):
        """
        Internal use only, not a feature.
        :return: True if exactly one field surrounding this space is free.
        """
        if x > 15 or x < 1 or y > 15 or y < 1:
            return False  # grid borders are not dead ends
        arena = self.arena
        free = (0, 3)  # walkable fields in arena
        bomb = (bomb_x, bomb_y)
        neighbors = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if bomb in neighbors:
            # check the case that player is completely trapped next to bomb (Sackgassentiefe 1)
            for n_x, n_y in neighbors:
                # if n_x > 16 or n_x < 0 or n_y > 16 or n_y < 0:
                #     continue
                if arena[n_x, n_y] in free:
                    return False
            return True

        # check if dead end is deeper
        return len([True for f in free if [arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(f) == 1 ]) == 1

    def _get_bomb_place_value_map(self):
        """
        Gives coordiantes and a weight map for every field of the arena. The smaller numbers are, the more this field is
        worth placing a bomb there. The output can be used to feed the _look_for_tarrgets_weighted function.
        :return: tuple(coordinates of the targets as [n,2] array, integer map of the arena's dimensions)
        """
        copied_arena = np.copy(self.arena)  # Copy as we will modify it
        result = np.zeros(copied_arena.shape)
        crate_ind = np.where(copied_arena == 1)
        crate_coords = np.vstack((crate_ind[0], crate_ind[1])).T
        x, y = self.player.me_loc[0], self.player.me_loc[1]

        for c in crate_coords:
            cx, cy = c[0], c[1]
            down, up, left, right = True, True, True, True
            for i in range(3):
                if down and copied_arena[cx, cy + (i + 1)] == -1: down = False
                if up and copied_arena[cx, cy - (i + 1)] == -1: up = False
                if left and copied_arena[cx - (i + 1), cy] == -1: left = False
                if right and copied_arena[cx + (i + 1), cy] == -1: right = False

                if down: result[cx, cy + (i + 1)] += 1
                if up: result[cx, cy - (i + 1)] += 1
                if left: result[cx - (i + 1), cy] += 1
                if right: result[cx + (i + 1), cy] += 1

            # As the break is removed, killing foes will contribute to fields' values as well
        # for c in self.player.foes:
        # print(result)

        copied_arena[x, y] = 0
        # if self.logger: self.logger.info(f'danger_map type: {type(self.danger_map)}')
        # if self.logger: self.logger.info(f'danger_map: {self.danger_map * 1}')
        result_coords = np.where((result > 0) & ((copied_arena == 0) | (copied_arena == 3)) & (self.danger_map))
        stacked_result_coords = np.vstack(result_coords).T

        weights = np.zeros([result_coords[0].shape[0]])
        for i in range(result_coords[0].shape[0]):
            weights[i] = result[stacked_result_coords[i, 0], stacked_result_coords[i, 1]]

        return stacked_result_coords, weights

    def d_best_bomb_dropping_dir(self):
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        # Switch off when 10 or less crates are on the field
        if self.player.foes.shape[0] == 0 or np.sum(self.arena == 1) > 10:  # FIXME no foes left
            target_coords, weights = self._get_bomb_place_value_map()
            free_space = (self.arena == 0) | (self.arena == 3)
            # if self.logger: 
            #     self.logger.info(f'TARGET COORDS, WEIGHTS: {target_coords, weights}')
            #     self.logger.info(f'arena: {self.arena}')
            free_space[x, y] = True
            best_step = self._look_for_targets_weighted(free_space, (x, y), target_coords, weights, None)
            return self._determine_direction(best_step, x, y)
        else:
            return self._determine_direction(None, x, y)


    def _name_player_events(self):
        """
        Debugging function to return events from this state.
        :return:
        """
        events = []

        for i in range(4):
            events.append([False for i in range(17)])
            for j, count in enumerate(self.state[self.board.shape[0] + 4 + i * 21: self.board.shape[0] + 4 + i * 21 + 17]):
                if count != 0:
                    events[i][j] = True

            events[i] = np.array(settings.events)[np.array(events[i])]
        return events

    def _get_reward(self, indices):
        """
        Debugging function to show player rewards
        :param indices:
        :return:
        """
        rewards = [get_reward(self.state, player_index=player) for player in indices]
        return rewards

class _Player:
    """
    Helper class to store useful attributes for a certain player.
    """

    def __init__(self, observation_self, player_index):
        """
        Setup basic members (but not distances to foes or coin_locs, use setup methods when creating features)
        :param observation_self: self member of Obervation Object
        :param player_index:
        """
        self.observation_self = observation_self
        self.player_index = player_index  # index of player in game state vector (0 to 3)
        self.me_loc = np.array([*index_to_x_y(observation_self.player_locs[int(player_index)])])

        self.coin_dists, self.closest_coin = None, None  # distances of all coin_locs, index of closest coin
        self.foes = np.array([foe_loc for ind, foe_loc in enumerate(observation_self.player_locs) if ind != player_index
                              and foe_loc != 0])  # count LIVING enemies
        self.foe_dists, self.closest_foe = None, None

        self._is_setup_coins, self._is_setup_foes = False, False

    def setup_coin_dists(self):
        """
        Set a list of coin distances and find closest coin (set to None if no coin_locs)
        :return:
        """
        if self._is_setup_coins:
            return  # don't calculate values twice

        self.coin_dists = np.array([np.linalg.norm(self.me_coords - np.array([*index_to_x_y(coin)]), ord=1)
                                      for coin in self.observation_self.coins])  # manhattan dist. to coin_locs
        self.closest_coin = np.argmin(self.coin_dists) if self.observation_self.coins.shape[0] != 0 else None

        self._is_setup_coins = True

    def setup_foe_dists(self):
        """
        Set a list of foe distances and find closest coin (set to None if no coin_locs)
        :return:
        """

        if self._is_setup_foes:
            return  # don't calculate values twice
        self.foe_dists = np.array([self.observation_self.player_distance_matrix[min(self.player_index, foe_index),
                                                                 max(foe_index, self.player_index)]
                                     for foe_index in range(4) if foe_index != self.player_index])
        # manhattan dist. to foes
        self.closest_foe = int(np.argmin(self.foe_dists)) if self.foes.shape[0] != 0 else None   # player index of closest foe

        if self.closest_foe is not None:
            self.closest_foe += 1 if self.closest_foe >= self.player_index else 0  # me not in foe_dists

        self._is_setup_foes = True
