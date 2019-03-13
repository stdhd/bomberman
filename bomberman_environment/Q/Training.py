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

            last_actions = these_actions.astype(int)

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


def update_and_get_obs(db, new_obs, learned, radius):
    radius = 2
    findings = np.where((db == new_obs).all(axis=1))
    if findings.shape[0] > 0:
        return db, findings[0], learned
    else:
        single = new_obs[:(radius * 2 + 1) ** 2 - 1]
        candidates = np.zeros(8, (radius * 2 + 1) ** 2)
        single_reshaped = np.reshape(single, (radius * 2 + 1, radius * 2 + 1))
        candidates[0] = single_reshaped
        candidates[1] = single_reshaped.T
        candidates[2] = np.flip(single_reshaped, 0)
        candidates[3] = np.flip(single_reshaped, 1)
        candidates[4] = np.flip(single_reshaped.T, 0)
        candidates[5] = np.flip(single_reshaped.T, 1)
        candidates[6] = np.fliplr(np.flip(single_reshaped, 0))
        candidates[7] = np.fliplr(np.flip(single_reshaped.T, 0))

        alternative = np.where((candidates == new_obs).all(axis=1))
        if alternative.shape[0] > 0:
            return db, alternative[0], learned
        else:
            learned = np.append(learned, np.zeros([1, learned.shape[1]]), axis=0)
            db = np.append(db, np.array([new_obs]), axis=0)
            return db, db.shape[0] - 1, learned
