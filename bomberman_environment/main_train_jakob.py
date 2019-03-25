

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

    obs = ObservationObject(0, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                'me_has_bomb',
                                'dead_end_detect',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d',
                                'd_best_bomb_dropping_dir',
                                'd_closest_enemy_dir'
                                ], None)

    write_path = 'data/qtables/' + obs.get_file_name_string()

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    KNOWN, Q = q_train_from_games_jakob(cwd + "/" + 'data/games/four_players_esa_0_2_cratedens_0_75',
                                        write_path,
                                        obs, a=0.8, g=0.7, save_every_n_files=5)


if __name__ == '__main__':
    main()