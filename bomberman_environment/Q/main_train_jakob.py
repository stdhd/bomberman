

from Q.Training import q_train_from_games_jakob

from agent_code.observation_object import ObservationObject

def main():
    """
    Runs and trains a Q-learning model.
    :return:
    """

    print("hello")

    KNOWN, Q = q_train_from_games_jakob('data/games', 'data/qtables/jakob', ObservationObject(2, ['closest_coin_dir']))

