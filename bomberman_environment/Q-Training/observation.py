import numpy as np
#from ..state_functions.rewards import *
#from ..indices import *

from rewards import *
from indices import *

import math


def create_observation(daten, radius, selected_agents):
    """
    Given a state file, the observation window radius and an array of
    the selected agents, returns the following vectors per agent: observation, action, reward

    :param daten: path and file name of the .npy file
    :param radius: integer value for the size of the observation window
    :return: observation: shape [agent_count, observation_size]
    :return: action: shape [agent_count]
    :return: reward: shape [agent_count]
    """

    ZEILEN = 17
    WALL = 2
    CRATE = 1
    FREE = 0
    COIN = -1
    PLAYER = 3
    BOMB = 4
    EXPLOSION = 5
    AGENT = 6
    INDEX_LENGTH = 176

    agents_count = selected_agents.shape[0]
    row_length = radius * 2 + 1
    array_target = np.zeros([agents_count, row_length, row_length])
    rewards = np.zeros([agents_count])
    action = np.zeros([agents_count])

    # Add Player and Bomb coordinates
    startX = INDEX_LENGTH
    l = 0
    for k in selected_agents:
        mx, my = index_to_x_y(int(daten[int(startX)]), ZEILEN,
                              ZEILEN)  # Koordinaten des Spielers, Mittelpunkt des Ausschnitts
        borderX = mx + radius
        borderY = my + radius

        start = INDEX_LENGTH
        for i in range(4):
            agentX, agentY = index_to_x_y(int(daten[int(start)]), ZEILEN, ZEILEN)
            # Set agents' positions
            if not (agentX > borderX or agentY > borderY or agentX < (mx - radius) or agentY < (my - radius)):
                array_target[l, int(agentY - (my - radius)), int(agentX - (mx - radius))] = AGENT
                start = start

            # Write bombs
            start += 2
            if not daten[int(start)] == 0:
                bombeX, bombeY = index_to_x_y(int(daten[int(start)]), ZEILEN, ZEILEN)
                if not (bombeX > borderX or bombeY > borderY or bombeX < (mx - radius) or bombeY < (my - radius)):
                    start = start
                    array_target[l, int(bombeY - (my - radius)), int(bombeX - (mx - radius))] = BOMB
            start = start + 19

        # Write rewards
        rewards[l] = get_reward(daten, l)

        # Write event
        action[l] = np.argmax(daten[int(startX + 4): int(startX + 8)])


        for i in range(int(math.pow(row_length, 2))):

            xt = i % row_length
            yt = math.floor(i / (row_length))
            x = xt + (mx - radius)
            y = yt + (my - radius)

            if array_target[l, int(yt), int(xt)] == 0:

                try:
                    index = x_y_to_index(int(x), int(y), ZEILEN, ZEILEN)
                    start = start
                    array_target[l, int(yt), int(xt)] = daten[int(index)]
                except Exception as e:
                    array_target[l, yt, xt] = WALL
                    start = start

        l += 1
        startX = startX + 21
        observation_vector = np.zeros([agents_count,array_target.shape[1] * array_target.shape[1]])
        for i in range(agents_count):
            observation_vector[i, :] = array_target[i].flatten()

    return observation_vector.astype(int), action.astype(int), rewards.astype(int)


def find_equivalent(single, collection, radius):
    """
    Given a single state vector (arena window + special features) and a array of multiple state vectors
    the function returns the index of the equivalent vector in the array by executing matrix transformations

    :param single: 1D observation vector
    :param collection: shape [observations_count, elements_per_observation]
    :return: index within collection or -1 if there is no equivalent
    """

    candidates = np.zeros((8, single.shape[0], single.shape[1]))
    single_reshaped = np.reshape(single,(radius*2 + 1, radius*2 + 1))
    candidates[0] = single_reshaped
    candidates[1] = single_reshaped.T
    candidates[2] = np.flip(single_reshaped, 0)
    candidates[3] = np.flip(single_reshaped, 1)
    candidates[4] = np.flip(single_reshaped.T, 0)
    candidates[5] = np.flip(single_reshaped.T, 1)
    candidates[6] = np.fliplr(np.flip(single_reshaped, 0))
    candidates[7] = np.fliplr(np.flip(single_reshaped.T, 0))

    for i in range(collection.shape[0]):
        for j in range(8):
            if (collection[i] == candidates[j]).all():
                return i

    return -1





#pass
#ff = np.load("test_daten.npy")
#for i in range(400):
  #  daten = np.load("test_daten.npy")[i]
    #get_reward(daten, 0)
 #   print(get_reward(daten, 0), get_reward(daten, 1), get_reward(daten, 2), get_reward(daten, 3))

#print(create_window(daten,1, np.array([0,1])))

