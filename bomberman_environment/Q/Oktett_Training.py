from os import listdir
from os.path import isfile, join

from agent_code.observation_object import ObservationObject
from state_functions.rewards import *
from Q.manage_training_data import *

def q_train_from_games_jakob(train_data, tables_logs_output_path, obs:ObservationObject, a = 0.8, g = 0.4, stop_after_n_files=None,
                             save_every_n_files:int=5):
    """
    Trains from all files in a directory using an existing q- and observation-table under tables_logs_output_path.

    If tables do not exist, creates them.

    Creates or updates json files indexing known training files under tables_logs_output_path.

    Notes training progress (steps seen and current size of Q Table) in json file.
    
    Uses preconfigured ObservationObject to train. 

    :param train_data: Directory containing training episodes
    :param tables_logs_output_path: Directory to which to output Q-learning results and json files.
    :param obs Observation Object containing training settings (view radius etc.)
    :param a alpha (learning rate)
    :param g gamma (discount)
    :param stop_after_n_files: Aborts training after training with this many files
    :param save_every_n_files: Count to this many files before saving (and then repeat)
    :return:
    """

    debug_mode = False
    if stop_after_n_files is not None and stop_after_n_files % save_every_n_files != 0:
        raise ValueError("Saves and stops should be multiples of each other")
    filename = obs.get_file_name_string()
    try:
        QTABLE = np.load(tables_logs_output_path + '/q_table-' + filename + '.npy')
        KNOWN = np.load(tables_logs_output_path + '/observation-' + filename + '.npy')
        QUANTITY = np.load(tables_logs_output_path + '/quantity-' + filename + '.npy')
    except:
        print("Error loading learned q table. using empty table instead.")
        QTABLE = np.zeros([0,6])
        QUANTITY = np.zeros([0,6])
        KNOWN = np.zeros([0, obs.obs_length])

    filecount = 0
    current_trained_batch = []  # keeps track of the files used for training
    steps_count = 0  # number of steps seen since launching training

    files = [f for f in listdir(train_data) if isfile(join(train_data, f))]

    for file in files:
        # go through files
        try:
            if is_trained(tables_logs_output_path + "/records.json", file):
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

        obs.reset_killed_players()  # renew information about killed players

        for ind, step_state in enumerate(game):

            obs.set_state(step_state)  # Set this first to initialize members
            living_players = obs.living_players
            just_died = obs.just_died  # find players that died in this step
            rewards_players = np.concatenate((living_players, just_died))

            last_actions = np.zeros(4)

            for player in range(4):
                actions_taken = step_state[int(obs.board.shape[0] + 4 + player * 21): int(obs.board.shape[0] + 9 + player * 21)]
                actions_taken = np.append(actions_taken, step_state[int(obs.board.shape[0] + 11 + player * 21)])
                last_actions[player] = np.argmax(actions_taken)

                if ind != 0 and player in living_players and actions_taken[np.where(actions_taken != 0)].shape[0] != 1:
                    print("Warning: Incorrect number of actions taken in one step for player index", player, "in step"
                    " number", ind, "in file", file)

            step_observations = obs.create_observation(living_players)

            terminal_states = np.zeros((just_died.shape[0], obs.obs_length))  # get null observations for died players

            for count, observation in enumerate(np.concatenate((step_observations, terminal_states))):

                if (count < living_players.shape[0] and np.array_equal(observation, np.zeros(obs.obs_length)))\
                        or (count >= living_players.shape[0] and not np.array_equal(observation, np.zeros(obs.obs_length))):
                    raise RuntimeError("Error: Got zero observation from living player or nonzero observation from "
                                       "dead player")

                findings = np.where((KNOWN == np.array([observation])).all(axis=1))[0]

                if findings.shape[0] > 0:  # Found observation in database

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
                    # KNOWN = np.concatenate(np.array([KNOWN, new_obs])) FIXME Changed to append
                    KNOWN = np.append(KNOWN, new_obs, axis=0)
                    QTABLE = np.append(QTABLE, np.zeros([n_new_indices, QTABLE.shape[1]]), axis=0)
                    QUANTITY = np.append(QUANTITY, np.zeros([n_new_indices, QTABLE.shape[1]]), axis=0)
                    index_current = np.arange(KNOWN.shape[0] - n_new_indices, KNOWN.shape[0])

                if ind > 0:
                
                    for i in range(last_index[rewards_players[count]][0].shape[0]):
                        l_ind, l_rot = last_index[rewards_players[count]][0][i], last_index[rewards_players[count]][1][i]
                        best_choice_current_state = np.max(QTABLE[int(index_current[0])])
                        # max of current state's Q Table values
                        reward = get_reward(step_state, rewards_players[count])
                        debug_helper = np.where(l_rot == last_actions[rewards_players[count]])[0]
                        if debug_helper.shape[0] == 0:
                            print()
                        rotated_action = np.where(l_rot == last_actions[rewards_players[count]])[0][0]
                        QTABLE[int(l_ind), rotated_action] = (1 - a) *  \
                        QTABLE[int(l_ind), rotated_action] +\
                        a * (reward + g * best_choice_current_state)

                        QUANTITY[int(l_ind), rotated_action] += 1

                last_index[rewards_players[count]] = (index_current, rotations_current)

            steps_count += 1

        if not KNOWN.shape[0] % 8 == 0:
            raise ValueError("Size of observation database must be product of 8*n")

        filecount += 1
        current_trained_batch.append(file)
        print("Trained with file", file)

        save_all = False
        stop = False

        if filecount % save_every_n_files == 0:
            save_all = True
        if stop_after_n_files is not None and filecount == stop_after_n_files:
            print("Stopping after", stop_after_n_files)
            save_all = True
            stop = True
        if filecount == len(files):
            print("Finished: Reached end of folder after", filecount, "files.")
            save_all = True
            stop = True

        if save_all:
            add_to_trained(tables_logs_output_path + "/records.json", current_trained_batch)  # update json records file
            catalogue_progress(tables_logs_output_path + "/progress.json", steps_count, QTABLE.shape[0])  # update json progress file

            np.save(tables_logs_output_path + '/observation-' + filename, KNOWN)
            np.save(tables_logs_output_path + '/q_table-' + filename, QTABLE)
            np.save(tables_logs_output_path + '/quantity-' + filename, QUANTITY)

            steps_count = 0
            current_trained_batch.clear()

        if stop:
            break

    return KNOWN, QTABLE


def get_transformations(obs, radius, direction_sensitive):
    direction_sensitive = np.array(direction_sensitive)
    if np.array_equal(obs, np.zeros(obs.shape[0])):
        # in case of terminal state, return zero vectors
        candidates = np.zeros([8, obs.shape[0]])
        direction_change = np.zeros([8, 6])
        return candidates, direction_change
    if radius == -1:
        new_window = np.array([])
    else:
        new_window = obs[:(radius * 2 + 1) ** 2]
        new_window_reshaped = new_window.reshape([radius * 2 + 1, radius * 2 + 1])
    new_rest = obs[new_window.shape[0]:]

    # All in all transformations, the direction sensitive features are replaced by the corresponding value 0-3
    # Non transformed data: 0=left, 1=right, 2=up, 3=down
    # Array positions are the values to replace: position 0 = normal left is replaced by the value in the array on
    # the 0. position

    # Read like 2 (up) --> 0 (down)
    transformations = np.array(
        [[2, 3, 0, 1], [1, 0, 2, 3], [0, 1, 3, 2], [3, 2, 0, 1], [2, 3, 1, 0], [1, 0, 3, 2], [3, 2, 1, 0]])

    all_transformed = np.zeros([7, new_rest.shape[0]])
    direction_change = np.zeros([8, 6])
    for i in range(7):
        all_transformed[i][new_rest == 0] = np.where(transformations[i] == 0)[0][0]
        all_transformed[i][new_rest == 1] = np.where(transformations[i] == 1)[0][0]
        all_transformed[i][new_rest == 2] = np.where(transformations[i] == 2)[0][0]
        all_transformed[i][new_rest == 3] = np.where(transformations[i] == 3)[0][0]
        all_transformed[i][new_rest == 4] = 4
        all_transformed[i][new_rest == 5] = 5
        all_transformed[i][new_rest == 6] = 6

    transformed_rest = np.zeros([7, new_rest.shape[0]])
    for i in range(7):
        transformed_rest[i] = np.where(direction_sensitive == 1, all_transformed[i], new_rest)
        feat2_transformed = np.zeros([4])
        to_skip = 0
        for start in range(new_rest.shape[0]):
            if direction_sensitive[start] == 2 and to_skip < 1:
                bit = transformed_rest[i,start:start + 4]
                feat2_transformed[0] = bit[transformations[i,0]]
                feat2_transformed[1] = bit[transformations[i,1]]
                feat2_transformed[2] = bit[transformations[i,2]]
                feat2_transformed[3] = bit[transformations[i,3]]
                transformed_rest[i,start:start + 4] = feat2_transformed
                to_skip = 3
            else:
                to_skip -= 1

    results = np.zeros([8, obs.shape[0]])
    if radius == -1:
        for i in range(7):
            results[i] = transformed_rest[i]
            direction_change[i] = np.append(transformations[i], np.array([4, 5]))
    else:
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

# def get_transformed_events(original_events):
#     original_events = np.array(original_events)
#     # Read like 2 (up) --> 0 (down)
#     # Diagonal (upper left to down right), vertical, horizontal, rotation right, rotation left, horizontal & vertical, Diagonal (down left to upper right)
#     transformations = np.array(
#         [[2, 3, 0, 1], [1, 0, 2, 3], [0, 1, 3, 2], [3, 2, 0, 1], [2, 3, 1, 0], [1, 0, 3, 2], [3, 2, 1, 0]])
#     transformed_events = np.empty([0, original_events.shape[0]])
#     for ind, trans in enumerate(transformations):
#         transformed_events = np.append(transformed_events, original_events[np.newaxis, :], axis=0)
#         transformed_events[ind][original_events == 0] = np.where(trans == 0)[0][0]
#         transformed_events[ind][original_events == 1] = np.where(trans == 1)[0][0]
#         transformed_events[ind][original_events == 2] = np.where(trans == 2)[0][0]
#         transformed_events[ind][original_events == 3] = np.where(trans == 3)[0][0]
#     transformed_events = np.append(transformed_events, original_events[np.newaxis, :], axis=0)
#     return transformed_events
