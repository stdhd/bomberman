from agent_code.merged_agent.indices import *
from Q.rewards import *
from os import listdir, getcwd
from os.path import isfile, join

from agent_code.observation_object import ObservationObject
from Q.manage_training_data import *

def q_train_from_games_jakob(train_data, write_path, obs:ObservationObject):
    """
    Trains from all files in a directory using an existing q- and observation-table under write_path.

    If tables do not exist, creates them.

    Creates json files indexing known training files under write_path.
    
    Uses preconfigured ObservationObject to train. 

    :param train_data: Directory containing training episodes
    :param write_path: Directory to which to output Q-learning results and json files.
    :param KNOWN Observation Object containing training settings (view radius etc.)
    :return:
    """

    try:
        QTABLE = np.load(write_path+"/learned.npy")
        KNOWN = np.load(write_path + "/observations.npy")
    except:
        print("Error loading learned q table. using empty table instead.")
        QTABLE = np.zeros([0,5])
        KNOWN = np.zeros([0, obs.obs_length])

    START = 176  # number of free grids in board

    a = 0.8  # alpha (learning rate)
    g = 0.8  # gamma (discount)

    for file in [f for f in listdir(train_data) if isfile(join(train_data, f))]:
        # go through files

        try:
            if is_trained(write_path+"/records.json", file):
                print("Skipping known training datum", file, "in folder", write_path)
                continue
        except IOError:
            print("Error accessing .json records for file", file, "in folder", train_data)

        game = np.load(train_data+"/"+file)

        these_actions = np.zeros(4)

        for ind, step_state in enumerate(game):

            last_actions = these_actions

            for player in range(4):
                these_actions[player] = np.argmax(step_state[int(START + 4 + player * 21): int(START + 8 + player * 21)])

            obs.set_state(step_state)

            living_players = np.arange(4)[np.where(obs.player_locs != 0)]

            step_observations = obs.create_observation(living_players)

            for observation in step_observations:
                KNOWN, index_current, QTABLE = update_and_get_obs(KNOWN, observation, QTABLE)
                best_choice_current_state = np.max(QTABLE[index_current])

            if ind > 0:
                for player_index in living_players:
                    QTABLE[last_index, last_actions[player_index]] = (1 - a) * QTABLE[last_index, last_actions[player_index]] + a * (get_reward(step_state, player_index) + g * best_choice_current_state)

            last_index = index_current

        np.save(write_path + "/observations.npy", KNOWN)
        np.save(write_path + "/learned.npy", QTABLE)

    return KNOWN, QTABLE

def q_train_from_games(reader_file, writer_path):
    """
    Reads all json-files from reader_path
    Trains Q table
    Writes resulting Q table to writer_path
    """
    AGENTS_COUNT = 4
    OBSERVATION_LENGTH = 25
    ACTIONS_COUNT = 6
    RADIUS = 2
    LEARNING_RATE = 0.8
    DISCOUNT = 0.8

    data = np.load(reader_file)
    learned = np.zeros([0, ACTIONS_COUNT])
    obs = np.zeros([0, OBSERVATION_LENGTH])
    action_db = np.zeros([data.shape[0], AGENTS_COUNT])

    for agent in range(AGENTS_COUNT):
        # Write events
        start = 176
        for i in range(data.shape[0]):
            action_db[i] = np.argmax(data[int(start + 4 + agent * 21): int(start + 8 + agent * 21)])

        for i in range(data.shape[0]):

            temp_observation = create_observation(data[i], RADIUS, agent)
            obs, index_current, learned = update_and_get_obs(obs, temp_observation, learned)
            my_best_value = np.max(learned[index_current])
            if i > 0:
                learned[last_index, action_db[agent, i]] = (1 - LEARNING_RATE) * learned[last_index, action_db[agent, i]] + LEARNING_RATE * (get_reward(data[agent,i-1]) + DISCOUNT * my_best_value)

            last_index = index_current
        np.save(writer_path + "/observation.npy",obs)
        np.save(writer_path + "/learned.npy", obs)
    return obs, learned


def update_and_get_obs(db, new_obs, learned):
    for i in range(db.shape[0]):
        if np.array_equal(db[i], new_obs):
            return db, i, learned
    db = np.append(db,np.array([new_obs]), axis = 0)
    learned = np.append(learned, np.zeros([1,learned.shape[1]]), axis = 0)

    return db, (db.shape[0] - 1), learned