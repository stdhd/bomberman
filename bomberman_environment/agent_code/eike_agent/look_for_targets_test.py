import numpy as np
import warnings
from random import shuffle
from time import time, sleep
from collections import deque
import os

def look_for_targets(free_space, start, targets, logger):
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
    while True:
        if current == start: print("Komisch", current)
        if parent_dict[current] == start: return current
        current = parent_dict[current]


free_space = np.array([[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
 [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False],
 [False,  True, False, False, False,  True, False,  True, False,  True, False, False,
  False,  True, False,  True, False],
 [False,  True,  True,  True,  True,  True,  True,  True,  True, False,  True,  True,
   True,  True,  True,  True, False],
 [False,  True, False,  True, False,  True, False,  True, False,  True, False,  True,
  False,  True, False,  True, False],
 [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
   True,  True,  True,  True, False],
 [False,  True, False,  True, False,  True, False,  True, False,  True, False,  True,
  False,  True, False,  True, False],
 [False,  True, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
   True,  True, False,  True, False],
 [False,  True, False,  True, False,  True, False,  True, False,  True, False,  True,
  False,  True, False,  True, False],
 [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
   True,  True,  True,  True, False],
 [False,  True, False,  True, False,  True, False,  True, False, False, False,  True,
  False,  True, False,  True, False],
 [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
   True,  True,  True,  True, False],
 [False, False, False,  True, False,  True, False,  True, False,  True, False, False,
  False,  True, False,  True, False],
 [False,  True,  True,  True,  True,  True,  True,  True, False,  True,  True,  True,
   True,  True,  True,  True, False],
 [False,  True, False,  True, False,  True, False,  True, False,  True, False,  True,
  False,  True, False,  True, False],
 [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
   True,  True,  True,  True, False],
 [False, False, False, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False]])

output = (3,11)
coins = np.array([[4, 3], [ 1,  9], [ 2, 11], [ 8,  1],[ 6, 9], [ 9,13], [15,  1], [11,  8], [14, 13]])
while(True):
    output = look_for_targets(free_space, output, coins, None)
    try:
      if np.where((coins == output).all(axis=1))[0].shape[0] != 0:
          coins = np.delete(coins, np.where((coins == output).all(axis=1))[0], axis=0)
          if(coins.shape[0] == 0):
            print(coins)
            break
    except:
      print("doNothing")
