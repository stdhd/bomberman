import numpy as np
import warnings
from random import shuffle
from time import time, sleep
from collections import deque
import os
from types import SimpleNamespace

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


def x_y_to_index(x, y):
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


def index_to_x_y(ind):
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


def _look_for_targets_safe_field(free_space, start, targets, logger):
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
    targets_not_reachable = True
    for t in targets:
        if (not tuple(t) in dist_so_far):
            continue
        else:
            targets_not_reachable = False
            break
    if targets_not_reachable:
        return 10
    while True:
        if current == start: print("Komisch", current)
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def _get_threat_map(self):
    """
    :return: Boolean map: True for Free/Coin, False for Wall/Threatened/Crate
    """
    arena_bool = (self.arena == 0) | (self.arena == 3)
    for loc in self.bomb_locs:
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

def _determine_direction(best_step, x, y):
        if best_step == (x-1,y): return 0 # move left
        if best_step == (x+1,y): return 1 # move right
        if best_step == (x,y-1): return 2 # move up
        if best_step == (x,y+1): return 3 # move down
        if best_step == None: return 4 # No targets exist.
        if best_step == (x,y): return 5 # Target can not be reached. Only occurs if current position is right before the obstacle.
        return 6 # Something else is wrong: This case should not occur

def d_closest_safe_field_dir(self):
        """
        Direction to next safe field.
        Bomb on arena: (16), 8, 4, 2
        Bomb and enemy on arena: 80, 40, 20, 10
        """
        x, y = self.player.me_loc[0], self.player.me_loc[1]
        # If there are no bombs on the field the direction should indicate this by turning off this feature (return 4)
        # if self.logger: self.logger.info(f'CHECK BOMBS: {self.bomb_locs, self.bomb_locs.any()}')
        if (not self.bomb_locs.any()): 
            # if self.logger: self.logger.info(f'NO BOMBS')
            return _determine_direction(None, x, y)
        arena = self.arena
        # If agent is not on danger zone indicate this by turning off feature (return 4)
        if self.danger_map[x, y]:
            # if self.logger: self.logger.info(f'NOT ON DANGER ZONE')
            return _determine_direction(None, x, y)
        free_space = (arena == 0) | (arena == 3)
        free_space_ind = np.where(self.danger_map == True)
        free_space_coords = np.vstack((free_space_ind[0], free_space_ind[1])).T
        best_step = _look_for_targets(free_space, (x, y), free_space_coords, None)
        # If not safe field is reachable search again with ignored explosion fields
        if best_step == 10:
            free_space = (arena == 0) | (arena == 3) | (arena == -2) | (arena == -4)
            danger_map_without_explosions = np.copy(free_space)
            for loc in self.bomb_locs:
                if loc == 0:
                    continue
                tx,ty = index_to_x_y(loc)
                danger_map_without_explosions[tx, ty] = False
                for i in range(3):
                    if self.arena[tx, ty + (i + 1)] == -1:
                        break
                    danger_map_without_explosions[tx, ty + (i + 1)] = False
                for i in range(3):
                    if self.arena[tx, ty - (i + 1)] == -1:
                        break
                    danger_map_without_explosions[tx, ty - (i + 1)] = False
                for i in range(3):
                    if self.arena[tx + (i + 1), ty] == -1:
                        break
                    danger_map_without_explosions[tx + (i + 1), ty] = False
                for i in range(3):
                    if self.arena[tx - (i + 1), ty] == -1:
                        break
                    danger_map_without_explosions[tx - (i + 1), ty] = False

            free_space_ind = np.where(danger_map_without_explosions == True)
            free_space_coords = np.vstack((free_space_ind[0], free_space_ind[1])).T
            best_step = _look_for_targets(free_space, (x, y), free_space_coords, None)
        # self.logger.info(f'XY_BOMBS: {np.vstack((x_bombs, y_bombs)).T}')
        # self.logger.info(f'Free Space Coords: {free_space_coords}')
        # self.logger.info(f'Self: {x, y}')
        # self.logger.info(f'Best_step: {best_step}')

        return _determine_direction(best_step, x, y)

arena = np.array(
 [[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  1., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1., -2., -1.,  1., -1.,  1., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  4.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1., -2., -1.,  1., -1.,  1., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.],
  [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
  [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])

# coins = np.array([[4, 3], [ 1,  9], [ 2, 11], [ 8,  1],[ 6, 9], [ 9,13], [15,  1], [11,  8], [14, 13]])
# coins = np.array([[1, 11]])
# crate = np.array([[1, 12]])

bombs = np.array([99,0,0,0])
player_pos = [7,9]
abcd = SimpleNamespace(
            bomb_locs=bombs, 
            arena=arena, 
            player=SimpleNamespace(me_loc = player_pos))

abcd.danger_map = _get_threat_map(abcd)

output = d_closest_safe_field_dir(abcd)
print(output)


# free_space = (arena == 0) | (arena == 3)
# output = look_for_targets(free_space, player_pos, coins, None)
# print(output)
# while(True):
#     output = look_for_targets(free_space, output, coins, None)
#     try:
#       if np.where((coins == output).all(axis=1))[0].shape[0] != 0:
#           coins = np.delete(coins, np.where((coins == output).all(axis=1))[0], axis=0)
#           if(coins.shape[0] == 0):
#             print(coins)
#             break
#     except:
#       print("doNothing")
