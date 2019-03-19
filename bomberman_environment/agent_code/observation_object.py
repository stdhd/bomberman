from state_functions.indices import *
from random import shuffle
import numpy as np
import settings


"""
Available Features: 

        me_has_bomb,
        closest_coin_dist,
        d_closest_coin_dir,
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
        d_closest_safe_field_dirNEW
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
        FEATURE_LIST = sorted(FEATURE_LIST, key=str.lower)
        self.features = FEATURE_LIST
        self.features.sort()

        self.radius = radius
        self.logger = logger

        self.window_size = (2 * radius + 1)
        self.NUM_FEATURES = len(self.features)
        self.obs_length = self.window_size ** 2 + self.NUM_FEATURES

        self.state, self.board, self.player_locs, self.coin_locs, self.player_distance_matrix = None, None, None, None, None

        self.arena = None

        self.bomb_locs, self.bomb_timers, self.died_players = None, None, None

        self.player = None  # helper variable for creating features (not observation window)

        self.name_dict = {
        "me_has_bomb": "mhb",
        "closest_coin_dist": "ccdist",
        "d_closest_coin_dir": "ccdir",
        "closest_foe_dist": "cfdist",
        "d_closest_foe_dir": "cfdir",
        "closest_foe_has_bomb": "cfhb",
        "nearest_foe_to_closest_coin": "nftcc",
        "dist_to_center": "dtc",
        "smallest_enemy_dist": "smd",
        "remaining_enemies": "re",
        "remaining_crates": "rcrates",
        "remaining_coins": "rcoins",
        "d_closest_safe_field_dir": "csfdir",
        "closest_coin_old": "cco",
        "d_closest_safe_field_dirNEW" : "csfdirN"

        }


    def set_state(self, state):

        """
        Set the state for the observation object and initialize useful feature values.
        :param state:
        :return:
        """

        self.state = state

        self._initialize_feature_helpers()

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

            window = self._make_window(radius, player_x, player_y)

            observations[count] = np.concatenate((window.flatten(), features[count]))  # concatenate window and features

        return observations

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
                    window[i, j] = self.board[x_y_to_index(lower_x + i, lower_y + j) - 1]

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
                    self.set_window(window, bomb_loc, center_x, center_y, radius_custom, 2)
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
        killed_booleans = np.array([player_blocks[i * 21 + 18] for i in range(4)])  # note "got killed" boolean
        self.died_players = np.where(killed_booleans >= 1)[0]
        # manhattan dist. to coin_locs
        self.arena = self._make_window(8, 8, 8)
        # self.player_distance_matrix = np.zeros((4, 4))
        # for p1 in np.arange(self.player_distance_matrix.shape[0]):
        #     for p2 in np.arange(start=p1 + 1, stop=self.player_distance_matrix.shape[1]):
        #         if self.player_locs[p1] == 0 or self.player_locs[p2] == 0:
        #             continue  # skip dead players
        #         self.player_distance_matrix[p1, p2] = np.linalg.norm(np.array([*index_to_x_y(self.player_locs[p1])])
        #                                                      - np.array([*index_to_x_y(self.player_locs[p2])]), ord=1)


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

    def _determine_direction(self, best_step, x, y):
        if best_step == (x-1,y): return 0 # move left
        if best_step == (x+1,y): return 1 # move right
        if best_step == (x,y-1): return 2 # move up
        if best_step == (x,y+1): return 3 # move down
        if best_step == None: return 4 # No targets exist.
        if best_step == (x,y): return 5 # Something is wrong: This case should not occur
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
        """
        arena = self.arena
        coins_ind = np.where(arena == 3)
        coins_coords = np.vstack((coins_ind[0], coins_ind[1])).T
        free_space = (arena == 0) | (arena == 3)
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        best_step = self._look_for_targets(free_space, (x, y), coins_coords, None)
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
        temp = np.array([])
        for f in self.features:
            if f.startswith("d_"):
                temp = np.append(np.array([True]), temp)
            else:
                temp = np.append(np.array([False]), temp)
        return temp

    def closest_coin_old(self):
        """
        Direction to player's nearest coin (outdated).
        :return:
        """
        player = self.player
        return self._get_path_dir(self.player_locs[player.player_index], self.player_locs[player.closest_coin])

    def d_closest_safe_field_dirNEW(self):
        """
        :return: Direction to take towards nearest field which is not threatened by bomb explosion
        """
        x, y = self.player.me_loc[0], self.player.me_loc[1]

        # Is agent already within safe zone?
        threat_map = self._get_threat_map()
        bool = threat_map[x,y]
        if not bool:
            targets = np.where(threat_map)
            target_coords = np.vstack((targets[0], targets[1])).T
            free_space = np.copy(threat_map)
            free_space[x, y] = True
            best_step = self._look_for_targets(free_space, (x, y), target_coords, None)
            if best_step == (x, y):
                return 4
            else:
                temp = self._determine_direction(best_step,x,y)
                return self._determine_direction(best_step,x,y)
        return 4 # By default return 4: There is no explosion threatening current field

    def _get_threat_map(self):
        """

        :return: Boolean map: True for Free/Coin, False for Wall/Threatened/Crate
        """
        arena = self.arena
        arena_bool = (arena == 0) | (arena == 3)
        for loc in self.bomb_locs:
            if loc == 0:
                continue
            tx,ty = index_to_x_y(loc, 17, 17)
            for i in range(4):
                if arena[tx, ty + i] == -1:
                    break
                arena_bool[tx, ty + i] = False
            for i in range(4):
                if arena[tx, ty - i] == -1:
                    break
                arena_bool[tx, ty - i] = False
            for i in range(4):
                if arena[tx + i, ty] == -1:
                    break
                arena_bool[tx + 1, ty] = False
            for i in range(4):
                if arena[tx - i, ty] == -1:
                    break
                arena_bool[tx - i, ty] = False
        return arena_bool

    def d_closest_safe_field_dir(self):
        """
        Direction to next safe field.
        Bomb on arena: (16), 8, 4, 2
        Bomb and enemy on arena: 80, 40, 20, 10
        """
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        # If there are no bombs on the field the direction should indicate this by turning off this feature (return 4)
        if not self.bomb_locs.any(): 
            if self.logger != None:
                self.logger.info(f'YES')
            return self._determine_direction(None, x, y)
        is_on_danger_zone_factor = 0
        arena = self.arena
        if self.logger != None:
            self.logger.info(f'ARENA: {arena}')
        danger_zone_coords = []
        down, up, left, right = True, True, True, True
        # for x_bomb, y_bomb in np.vstack((x_bombs, y_bombs)).T:
        for bomb_loc in self.bomb_locs:
            if bomb_loc != 0:
                x_bomb, y_bomb = index_to_x_y(bomb_loc)
                danger_zone_coords.append([x_bomb, y_bomb])
                for i in range(3):
                    if down and arena[x_bomb, y_bomb + (i + 1)] == -1: down = False
                    if up and arena[x_bomb, y_bomb - (i + 1)] == -1: up = False
                    if left and arena[x_bomb - (i + 1), y_bomb] == -1: left = False
                    if right and arena[x_bomb + (i + 1), y_bomb] == -1: right = False

                    if down: danger_zone_coords.append([x_bomb, y_bomb + (i + 1)]) 
                    if up: danger_zone_coords.append([x_bomb, y_bomb - (i + 1)])
                    if left: danger_zone_coords.append([x_bomb - (i + 1), y_bomb])
                    if right: danger_zone_coords.append([x_bomb + (i + 1), y_bomb])
        
        # If agent is on danger zone indicate this by turning off feature (return 4)
        if [x,y] not in danger_zone_coords:
            return self._determine_direction(None, x, y)
        danger_zone_coords = np.array(danger_zone_coords)
        free_space = (arena == 0) | (arena == 3)
        free_space_calc = np.copy(free_space)
        if danger_zone_coords.shape[0] != 0:
            free_space_calc[danger_zone_coords[:,0], danger_zone_coords[:, 1]] = False
        free_space_ind = np.where(free_space_calc == True)
        free_space_coords = np.vstack((free_space_ind[0], free_space_ind[1])).T

        best_step = self._look_for_targets(free_space, (x, y), free_space_coords, None)
        # self.logger.info(f'XY_BOMBS: {np.vstack((x_bombs, y_bombs)).T}')
        # self.logger.info(f'Danger Zone Coords: {danger_zone_coords}')
        # self.logger.info(f'Self: {x, y}')
        # self.logger.info(f'Best_step: {best_step}')

        return self._determine_direction(best_step, x, y)

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

        self.coin_dists = np.array([np.linalg.norm(self.me_loc - np.array([*index_to_x_y(coin)]), ord=1)
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
