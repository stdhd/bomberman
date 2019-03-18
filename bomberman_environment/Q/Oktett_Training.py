from agent_code.merged_agent.indices import *
from os import listdir, getcwd
from os.path import isfile, join

from agent_code.observation_object import ObservationObject
from state_functions.rewards import *
from Q.manage_training_data import *

def q_train_from_games_jakob(train_data, write_path, obs:ObservationObject, a = 0.8, g = 0.8):
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
    :return:
    """

    debug_mode = False
    filename = obs.get_file_name_string()
    try:
        QTABLE = np.load(write_path + '/q_table-' + filename + '.npy')
        KNOWN = np.load(write_path + '/observation-' + filename + '.npy')
        QUANTITY = np.load(write_path + '/quantity-' + filename + '.npy')
    except:
        print("Error loading learned q table. using empty table instead.")
        QTABLE = np.zeros([0,6])
        QUANTITY = np.zeros([0,6])
        KNOWN = np.zeros([0, obs.obs_length])

    for file in [f for f in listdir(train_data) if isfile(join(train_data, f))]:
        # go through files

        try:
            if is_trained(write_path+"/records.json", file):
                print("Skipping known training datum", file, "in folder", train_data)
                continue
        except IOError:
            print("Error accessing .json records for file", file, "in folder", train_data)

        try:
            game = np.load(train_data+"/"+file)
        except OSError:
            print("Skipping " + file + ". Is it a .npy file?")
            continue


        last_index = [None, None, None, None]

        for ind, step_state in enumerate(game):

            obs.set_state(step_state)
            living_players = np.arange(4)[np.where(obs.player_locs != 0)]
            last_actions = np.zeros(4)

            for player in range(4):
                actions_taken = step_state[int(obs.board.shape[0] + 4 + player * 21): int(obs.board.shape[0] + 9 + player * 21)]
                actions_taken = np.append(actions_taken, step_state[int(obs.board.shape[0] + 11 + player * 21)])
                last_actions[player] = np.argmax(actions_taken)

                if ind != 0 and player in living_players and actions_taken[np.where(actions_taken != 0)].shape[0] != 1:
                    print("Warning: Incorrect number of actions taken in one step for player index", player, "in step"
                    " number", ind, "in file", file)

            step_observations = obs.create_observation(living_players)

            window = obs._make_window(8, 8, 8)

            for count, observation in enumerate(step_observations):

                findings = np.where((KNOWN == np.array([observation])).all(axis=1))[0]

                if findings.shape[0] > 0:

                    candidates, rotations_current = get_transformations(observation, obs.radius,
                                                     obs.get_direction_sensitivity())

                    index_current = np.array([])
                    for cand in candidates:
                        temp = np.where((KNOWN == np.array([cand])).all(axis=1))[0]
                        if temp.shape[0] > 0:
                            index_current = np.append(index_current, temp[0])

                else:
                    new_obs, rotations_current = get_transformations(observation, obs.radius, obs.get_direction_sensitivity())
                    n_new_indices = new_obs.shape[0]
                    KNOWN = np.concatenate(np.array([KNOWN, new_obs]))
                    QTABLE = np.append(QTABLE, np.zeros([n_new_indices, QTABLE.shape[1]]), axis=0)
                    QUANTITY = np.append(QUANTITY, np.zeros([n_new_indices, QTABLE.shape[1]]), axis=0)
                    index_current = np.arange(KNOWN.shape[0] - n_new_indices, KNOWN.shape[0])

                if ind > 0:
                    for i in range(last_index[living_players[count]][0].shape[0]):
                        l_ind, l_rot = last_index[living_players[count]][0][i], last_index[living_players[count]][1][i]
                        best_choice_current_state = np.max(QTABLE[int(index_current[0])])
                        reward = get_reward(step_state, living_players[count])
                        rotated_action = np.where(l_rot == last_actions[living_players[count]])[0][0]
                        QTABLE[int(l_ind), rotated_action] = (1 - a) *  \
                        QTABLE[int(l_ind), rotated_action] +\
                        a * (reward + g * best_choice_current_state)

                        QUANTITY[int(l_ind), rotated_action] += 1

                        if i == 0 and reward != -3 and debug_mode:
                            print("-----")
                            print("DID: " + str(rotated_action))
                            print("Reward: " + str(reward))

                last_index[living_players[count]] = (index_current, rotations_current)

        add_to_trained(write_path+"/records.json", file)  # update json table

        print("Trained with file", file)

        if not KNOWN.shape[0] % 8 == 0:
            raise ValueError("Size of observation database must be product of 8*n")

        np.save(write_path + '/observation-' + filename, KNOWN)
        np.save(write_path + '/q_table-' + filename, QTABLE)
        np.save(write_path + '/quantity-' + filename, QUANTITY)

    return KNOWN, QTABLE


def get_transformations(obs, radius, direction_sensitive):
    new_window = obs[:(radius * 2 + 1) ** 2]
    new_rest = obs[new_window.shape[0]:]
    new_window_reshaped = new_window.reshape([radius * 2 + 1, radius * 2 + 1])

    # All in all transformations, the direction sensitive features are replaced by the corresponding value 0-3
    # Non transformed data: 0=left, 1=right, 2=up, 3=down
    # Array positions are the values to replace: position 0 = normal left is replaced by the value in the array on
    # the 0. position

    transformations = np.array(
        [[2, 3, 0, 1], [1, 0, 2, 3], [0, 1, 3, 2], [3, 2, 0, 1], [2, 3, 1, 0], [1, 0, 3, 2], [3, 2, 1, 0]])

    all_transformed = np.zeros([7, new_rest.shape[0]])
    direction_change = np.zeros([8, 6])
    for i in range(7):
        all_transformed[i][new_rest == 0] = transformations[i, 0]
        all_transformed[i][new_rest == 1] = transformations[i, 1]
        all_transformed[i][new_rest == 2] = transformations[i, 2]
        all_transformed[i][new_rest == 3] = transformations[i, 3]
        all_transformed[i][new_rest == 4] = 4
        all_transformed[i][new_rest == 5] = 5

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
        direction_change[i] = np.append(transformations[i], np.array([4, 5]))

    direction_change[7] = np.array([0,1,2,3,4,5])
    results[7] = obs

    return results, direction_change
