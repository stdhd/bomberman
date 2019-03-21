

from Q.Oktett_Training import q_train_from_games_jakob
from agent_code.observation_object import ObservationObject
from evaluation_environment import EvaluationEnvironment
import main_save_states
from settings import s
from settings_agent_evaluation import s as sae_s


def main(data_path=None, train_iterations = 10, train_batch_size = 10): # ='data/games/four_players_esa_0_2_cratedens_0_75/'):
    """
    Train an agent from the ground up and evaluate their performance every few games.
    Saves all files in a subdirectory of the agent's folder.

    Supports training + evaluation from pre-existing data (e.g. simple agents), but also training through self-play.
    :param data_path: If not None, take training data from here (should contain multiples of 100 games) -> Else perform
    self-play to learn
    :param train_iterations: How many training cycles to go through
    :param train_batch_size: How many files to train from in one cycle
    :return:
    """

    obs = ObservationObject(1, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                'd_closest_crate_dir',
                                'me_has_bomb',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d',
                                'dead_end_detect',
                                ], None)
    if data_path is None:
        create_data = True
        data_path = "data/games/SELFPLAY"+obs.get_file_name_string()

    else:
        create_data = False

    write_path = 'data/qtables/' + obs.get_file_name_string() + "/evaluategames"

    for i in range(train_iterations):

        iteration_str = "/ITERATION_"+str(i+1)

        if create_data:

            data_path = data_path+iteration_str

            training_setup = [
            ('testing_only', False),
            ('testing_only', False),
            ('testing_only', False),
            ('testing_only', False)
            ]

            print("Creating", s.n_rounds, "training episodes by playing agent against itself")

            main_save_states.main(training_setup, data_path)  # Important: Set settings.py nrounds to train batch size

        if create_data:
            print("Training from", train_batch_size, "newly played games.")
        else:
            print("Training from", train_batch_size, "games in", data_path)

        q_train_from_games_jakob(data_path, "data/qtables/"+obs.get_file_name_string(),
                                        obs, a=0.5, g=0.5, stop_after_n_files=train_batch_size, save_every_n_files=5)

        iter_output = write_path + iteration_str

        env = EvaluationEnvironment(["testing_only"], iter_output)

        print("Running", sae_s.n_rounds, "trials vs. simple agents")

        env.run_trials()

        env.analyze_games()


if __name__ == '__main__':
    main()

