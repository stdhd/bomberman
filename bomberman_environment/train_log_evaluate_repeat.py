

from Q.Oktett_Training import q_train_from_games_jakob
from agent_code.observation_object import ObservationObject
from evaluation_environment import EvaluationEnvironment


def main(data_path="data/games/"):
    """
    Train an agent from the ground up and evaluate their performance every few games.
    Saves all files in the agent's folder.
    :param data_path: If not None, take training data from here (should contain multiples of 100 games)
    :return:
    """

    if data_path is None:
        raise NotImplemented

    obs = ObservationObject(3, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                'me_has_bomb',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d'
                                ], None)

    write_path = 'data/qtables/' + obs.get_file_name_string() + "/evaluategames"

    train_iterations = 6

    train_batch_size = 100  # how many files to train with before testing

    for i in range(train_iterations):

        q_train_from_games_jakob('data/games/four_players_esa_0_2_cratedens_0_75/', write_path,
                                        obs, a=0.5, g=0.5, stop_after_n_files=train_batch_size)

        iter_output = write_path + "/ITERATION_1_"+str((i+1) * train_batch_size)

        env = EvaluationEnvironment(["testing_only"], iter_output)

        env.run_trials()

        env.analyze_games()


