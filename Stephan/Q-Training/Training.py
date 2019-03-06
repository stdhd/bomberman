import numpy as np
from random import *
import matplotlib.pyplot as plt
from indices import *
# from create_observation_V6 import *
import math


def q_train_from_games(reader_path, writer_path):
    """
    Reads all json-files from reader_path
    Trains Q table
    Writes resulting Q table to writer_path
    TODO: iterate over games = state files
    """
    AGENTS_COUNT = 4
    OBSERVATION_LENGTH = 8
    ACTIONS_COUNT = 6

    daten = np.load(reader_path)
    learned = np.zeros([daten, 6])
    obs = np.zeros([0, OBSERVATION_LENGTH])
    observation_db = np.zeros([daten.shape[0], AGENTS_COUNT, OBSERVATION_LENGTH])
    action_db = np.zeros([daten.shape[0], AGENTS_COUNT])
    reward_db = np.zeros([daten.shape[0], AGENTS_COUNT])

    for i in range(daten.shape[0]):
        temp_observation, temp_action, temp_reward = create_observation_vectors(daten[i], 2, AGENTS_COUNT)
        for j in range(AGENTS_COUNT):
            observation_db[i, j], action_db[i, j], reward_db[i, j] = temp_observation[j], temp_action[j], temp_reward[j]

    for k in range(AGENTS_COUNT):
        learned, obs = do_action(learned, obs, 0.8, 0.7, observation_db[:, j], action_db[:, j], reward_db[:, j])


def do_action(learned, obs, learning_rate, discount, observation_db, action_db, reward_db):
    last_action_index = -1
    last_index = -1
    last_reward = -1
    for i in range(observation_db.shape[0]):

        current_obs = observation_db[i]
        obs, index_current, learned = update_and_get_obs(obs, current_obs, learned)

        choice = action_db[i]

        my_best_value = np.max(learned[index_current])
        n_state = get_following_state(state, spielfeld, actions[choice], actions, True)
        if (not last_index == -1):
            learned[last_index, last_action_index] = (1 - learning_rate) * learned[
                last_index, last_action_index] + learning_rate * (last_reward + discount * my_best_value)

        last_reward = reward_db[i]
        last_action_index = choice
        last_index = index_current

    reihe = np.append(reihe, np.array([reihenvalue]))
    return learned, obs


def update_and_get_obs(db, new_obs, learned):
    temp = -1
    for i in range(db.shape[0]):
        if np.array_equal(db[i], new_obs):
            return db, i, learned
    db = np.append(db,np.array([new_obs]), axis = 0)
    learned = np.append(learned, np.zeros([1,learned.shape[1]]), axis = 0)
    return db, (db.shape[0] - 1), learned