from agent_code.merged_agent.indices import *
from os import listdir, getcwd
from os.path import isfile, join

from agent_code.observation_object import ObservationObject
from state_functions.rewards import *
from Q.manage_training_data import *

def q_train_from_games_jakob(train_data, write_path, obs:ObservationObject, a = 0.8, g = 0.8, START = 176):
    """
    Trains from all files in a directory using an existing q- and observation-table under write_path.

    If tables do not exist, creates them.

    Creates json files indexing known training files under write_path.
    
    Uses preconfigured ObservationObject to train. 

    :param train_data: Directory containing training episodes
    :param write_path: Directory to which to output Q-learning results and json files.
    :param obs Observation Object containing training settings (view radius etc.)
    :param a alpha (learning rate)
    :param g gamma (discount)
    :param START number of free grids in board
    :return:
    """

    filename = obs.get_file_name_string()
    try:
        QTABLE = np.load(write_path + '/q_table-' + filename)
        KNOWN = np.load(write_path + '/observation-' + filename)
    except:
        print("Error loading learned q table. using empty table instead.")
        QTABLE = np.zeros([0,6])
        KNOWN = np.zeros([0, obs.obs_length])

    for file in [f for f in listdir(train_data) if isfile(join(train_data, f))]:
        # go through files

        try:
            if is_trained(write_path+"/records.json", file):
                print("Skipping known training datum", file, "in folder", write_path)
                continue
        except IOError:
            print("Error accessing .json records for file", file, "in folder", train_data)

        try:
            game = np.load(train_data+"/"+file)
        except OSError:
            print("Skipping " + file + ". Is it a .npy file?")
            continue

        these_actions = np.zeros(6)

        for ind, step_state in enumerate(game):

            last_actions = these_actions.astype(int)
            pass
            for player in range(4):
                actions_taken = step_state[int(START + 4 + player * 21): int(START + 9 + player * 21)]
                actions_taken = np.append(actions_taken, step_state[int(START + 11 + player * 21)])
                these_actions[player] = np.argmax(actions_taken)

            obs.set_state(step_state)

            living_players = np.arange(4)[np.where(obs.player_locs != 0)]
            step_observations = obs.create_observation(living_players)

            for player_index, observation in enumerate(step_observations):

                findings = np.where((KNOWN == np.array([observation])).all(axis=1))[0]
                if findings.shape[0] > 0:

                    candidates = get_transformations(observation, obs.radius,
                                                     obs.get_direction_sensitivity())

                    index_current = np.array([])
                    for cand in candidates:
                        temp = np.where((KNOWN == np.array([cand])).all(axis=1))[0]
                        if temp.shape[0] > 0:
                            index_current = np.append(index_current, temp[0])

                else:
                    new_obs = get_transformations(observation, obs.radius, obs.get_direction_sensitivity())
                    n_new_indices = new_obs.shape[0]
                    KNOWN = np.concatenate(np.array([KNOWN, new_obs]))
                    QTABLE = np.append(QTABLE, np.zeros([n_new_indices, QTABLE.shape[1]]), axis=0)
                    index_current = np.arange(KNOWN.shape[0] - n_new_indices, KNOWN.shape[0])

            if ind > 0:
                #print(index_current)
                for player_index in living_players:
                    for l_ind in last_index:

                        best_choice_current_state = np.max(QTABLE[int(index_current[0])])
                        #print(l_ind)
                        QTABLE[int(l_ind), int(last_actions[player_index])] = (1 - a) * QTABLE[int(l_ind), int(last_actions[player_index])] + a * (get_reward(step_state, player_index) + g * best_choice_current_state)

            last_index = index_current

        add_to_trained(write_path+"/records.json", file)  # update json table

        print("Trained with file", file)

        np.save(write_path + '/observation-' + filename, KNOWN)
        np.save(write_path + '/q_table-' + filename, QTABLE)

    return KNOWN, QTABLE


def get_transformations(obs, radius, direction_sensitive):
    new_window = obs[:(radius * 2 + 1) ** 2]
    new_rest = obs[new_window.shape[0]:]
    new_window_reshaped = new_window.reshape([radius * 2 + 1, radius * 2 + 1])

    # All in all transformations, the direction sensitive features are replaced by the corresponding value 1-4
    # Non transformed data: 1=up, 2=right, 3=down, 4=left
    # Array positions are the values to replace: position 0 = normal up is replaced by the value in the array on
    # the 0. position

    transformations = np.array(
        [[3, 2, 1, 0], [2, 1, 0, 3], [0, 3, 2, 1], [3, 0, 1, 2], [1, 2, 3, 0], [2, 3, 0, 1], [1, 0, 3, 2]])

    all_transformed = np.zeros([7, new_rest.shape[0]])

    for i in range(7):
        all_transformed[i][new_rest == 0] = transformations[i, 0]
        all_transformed[i][new_rest == 1] = transformations[i, 1]
        all_transformed[i][new_rest == 2] = transformations[i, 2]
        all_transformed[i][new_rest == 3] = transformations[i, 3]

    transformed_rest = np.zeros([7, new_rest.shape[0]])
    results = np.zeros([8, obs.shape[0]])
    for i in range(7):
        transformed_rest[i] = np.where(direction_sensitive, all_transformed[i], new_rest)

    candidates = np.zeros([7, radius * 2 + 1, radius * 2 + 1])
    candidates[0] = new_window_reshaped.T
    candidates[1] = np.flip(new_window_reshaped, 0)
    candidates[2] = np.flip(new_window_reshaped, 1)
    candidates[3] = np.flip(new_window_reshaped.T, 0)
    candidates[4] = np.flip(new_window_reshaped.T, 1)
    candidates[5] = np.fliplr(np.flip(new_window_reshaped, 0))
    candidates[6] = np.fliplr(np.flip(new_window_reshaped.T, 0))

    for i in range(7):
        results[i] = np.append(candidates[i], transformed_rest[i])
    results[7] = obs

    return np.unique(results, axis=0)

def update_and_get_obs(db, new_obs, learned):
    findings = np.where((db == np.array([new_obs])).all(axis=1))[0]
    if findings.shape[0] > 0:
        return db, findings[0], learned
    else:
        learned = np.append(learned, np.zeros([1, learned.shape[1]]), axis = 0)
        db = np.append(db, np.array([new_obs]), axis=0)
        return db, db.shape[0] - 1, learned



def update_and_get_obs_new(db, new_obs, learned, observation_object, direction_sensitive):
    """
    Seach an observation table for a (potentially rotated) match for a given new observation.

    If a match is found, return the index in the table, as well as a rotation encoding.
    :param db: Database of known observations
    :param new_obs: New observation to find in the database
    :param learned: Q Table
    :param observation_object: Object containing observation info (use to derive radius)
    :param direction_sensitive: FIXME
    :return: updated database, index of observation, q table, rotation encoding
    """
    findings = np.where((db == np.array([new_obs])).all(axis=1))[0]

    if findings.shape[0] > 0:
        # print("Found without transformation")
        return db, findings[0], learned, np.array([1, 2, 3, 4])

    # Observation not found in collection of observations, so try to find a transformation:

    radius = observation_object.radius
    new_window = new_obs[:(radius * 2 + 1) ** 2]
    new_rest = new_obs[new_window.shape[0]:]
    new_window_reshaped = new_window.reshape([radius * 2 + 1, radius * 2 + 1])

    # All in all transformations, the direction sensitive features are replaced by the corresponding value 1-4
    # Non transformed data: 1=up, 2=right, 3=down, 4=left
    # Array positions are the values to replace: position 0 = normal up is replaced by the value in the array on
    # the 0. position
    transformations = np.array(
        [[3,2,1,0], [2, 1, 0, 3], [0, 3, 2, 1], [3, 0, 1, 2], [1, 2, 3, 0], [2, 3, 0, 1], [1, 0, 3, 2]])

    all_transformed = np.zeros([7, new_rest.shape[0]])

    for i in range(7):
        all_transformed[i][new_rest == 1] = transformations[i, 0]
        all_transformed[i][new_rest == 2] = transformations[i, 1]
        all_transformed[i][new_rest == 3] = transformations[i, 2]
        all_transformed[i][new_rest == 4] = transformations[i, 3]

    transformed_rest = np.zeros([7, new_rest.shape[0]])
    for i in range(7):
        transformed_rest[i] = np.where(direction_sensitive, all_transformed[i], new_rest)

    candidates = np.zeros([7, radius * 2 + 1, radius * 2 + 1])
    candidates[0] = new_window_reshaped.T
    candidates[1] = np.flip(new_window_reshaped, 0)
    candidates[2] = np.flip(new_window_reshaped, 1)
    candidates[3] = np.flip(new_window_reshaped.T, 0)
    candidates[4] = np.flip(new_window_reshaped.T, 1)
    candidates[5] = np.fliplr(np.flip(new_window_reshaped, 0))
    candidates[6] = np.fliplr(np.flip(new_window_reshaped.T, 0))

    for i in range(7):
        alternative = np.append(candidates[i], transformed_rest[i])
        findings = np.where((db == np.array([alternative])).all(axis=1))[0]
        if findings.shape[0]:
            # Found transformed alternative
            # print("Found as transformation")
            return db, findings[0], learned, transformations[i]

    learned = np.append(learned, np.zeros([1, learned.shape[1]]), axis=0)
    db = np.append(db, np.array([new_obs]), axis=0)
    # print("Not Found")
    return db, db.shape[0] - 1, learned, np.array([1, 2, 3, 4])
