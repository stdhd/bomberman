

from Q.Oktett_Training import q_train_from_games_jakob

import os

from agent_code.observation_object import ObservationObject


def main():
    """
    Runs and trains a Q-learning model.
    :return:
    """
    os.chdir(os.path.dirname(__file__))
    cwd = os.getcwd()

    obs = ObservationObject(2, None, ['d_closest_coin_dir'])

    write_path = 'data/qtables/' + obs.get_file_name_string()
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    KNOWN, Q = q_train_from_games_jakob(cwd + "/" + 'data/games/one_player_only_coins/', write_path,
                                        obs)


if __name__ == '__main__':
    main()