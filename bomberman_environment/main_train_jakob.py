

from Q.Training import q_train_from_games_jakob

import os

from agent_code.observation_object import ObservationObject


def main():
    """
    Runs and trains a Q-learning model.
    :return:
    """
    os.chdir(os.path.dirname(__file__))
    cwd = os.getcwd()

    KNOWN, Q = q_train_from_games_jakob(cwd + "/" + 'data/games', 'data/qtables/jakob', ObservationObject(2, ['dist_to_center']))


if __name__ == '__main__':
    main()