

import matplotlib.pyplot as plt
import json
import numpy as np

from agent_code.observation_object import ObservationObject


def main():
    """
    Load and shows a progress file.
    :return:
    """

    obs0 = ObservationObject(0, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                'd_best_bomb_dropping_dir',
                                'me_has_bomb',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d',
                                'dead_end_detect',
                                #'d_closest_crate_dir',
                                #'d_closest_enemy_dir'
                                ], None)

    obs1 = ObservationObject(1, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                'd_best_bomb_dropping_dir',
                                'me_has_bomb',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d',
                                'dead_end_detect',
                                # 'd_closest_crate_dir',
                                'd_closest_enemy_dir'
                                ], None)

    obs3 = ObservationObject(3, ['d_closest_coin_dir',
                                'd_closest_safe_field_dir',
                                #'d_best_bomb_dropping_dir',
                                'me_has_bomb',
                                'd4_is_safe_to_move_a_l',
                                'd4_is_safe_to_move_b_r',
                                'd4_is_safe_to_move_c_u',
                                'd4_is_safe_to_move_d_d',
                                #'dead_end_detect',
                                # 'd_closest_crate_dir',
                                #'d_closest_enemy_dir'
                                ], None)

    obss = [obs0, obs1, obs3]
    feats = [(0, 6, 3), (1, 6, 4), (3, 5, 2)]

    for ind, obs in enumerate(obss):

        filepath = "data/qtables/" + obs.get_file_name_string()+"/"

        # db = view_db(filepath+"observation-"+obs.get_file_name_string()+".npy")

        set_progress_chart(filepath)

        radius, binary, dir = feats[ind]

        title = "Radius "+str(radius) + " with " + str(binary) + " binary features and " + str(dir) + \
                " directional features"

        plt.title(title)
        plt.xlabel("Number of steps seen")
        plt.ylabel("Length of Q Table")

        plt.show()

        plt.title(title)
        set_quantities_histogram(filepath + "quantity-" + obs.get_file_name_string() + ".npy")
        plt.xlabel("Observation frequency")
        plt.ylabel("Count")

        plt.show()

def view_db(filepath:str):
    """
    Show the obs db (debugging purposes)
    :param filepath:
    :return:
    """

    db = np.load(filepath)

    return db

def set_progress_chart(filepath:str):
    """
    Sets a progress chart from progress .json record
    :param filepath:
    :return:
    """
    with open(filepath+"progress.json", "r") as f:
        x, y = json.load(f)

    for i, val in enumerate(x):
        if i > 0:
            x[i] += x[i-1]

    x = np.array(x)
    plt.plot(x, y)

    return x, y


def set_quantities_histogram(filepath:str):
    """
    Sets a histogram to pyplot from quantities array
    :param filepath:
    :return:
    """

    q = np.load(filepath)

    q = q.sum(axis=1).flatten()

    num_bins = 70

    n, bins, patches = plt.hist(q, num_bins, facecolor='blue', alpha=0.5, log=True)

    # plt.plot(bins[:-1], n)

if __name__ == '__main__':
    main()

