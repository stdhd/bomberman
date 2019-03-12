from agent_code.marathon.indices import *
from Q.create_observation_jakob import *
from Q.rewards import *



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