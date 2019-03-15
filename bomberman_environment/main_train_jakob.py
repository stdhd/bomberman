

from Q.Training import q_train_from_games

import os

from agent_code.observation_object import ObservationObject


def main():
    """
    Runs and trains a Q-learning model.
    :return:
    """
    os.chdir(os.path.dirname(__file__))
    cwd = os.getcwd()

    KNOWN, Q = q_train_from_games(cwd + "/" + 'data/games/coins_only_one_player', 'data/qtables/jakob',
                                  ObservationObject(2, None, ['closest_coin_old']))


if __name__ == '__main__':
    main()