

import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter

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

    #f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
    #axes = (ax1, ax2, ax3)
    plt.rcParams.update({'font.size': 16})
    #matplotlib.rc('xtick', labelsize=22)
    #matplotlib.rc('ytick', labelsize=22)
    labelsize = 16
    #f.suptitle('Effects of view window size on Q Table length')

    #form = ScalarFormatter(style='sci', scilimits=(-2, 20))

    for ind, obs in enumerate(obss):

        filepath = "data/qtables/" + obs.get_file_name_string()+"/"

        if ind == 0:
            filepath = filepath[:-1] + "OLD/"  #

        # db = view_db(filepath+"observation-"+obs.get_file_name_string()+".npy")
        
        #ax = axes[ind]



        radius, binary, dir = feats[ind]

        title = "Radius "+str(radius) + " with " + str(binary) + " binary features \nand " + str(dir) + \
                " directional features"

        # ax.set_title(title)
        # ax.set_xlabel("Number of steps seen")
        # if ind == 0:
        #     ax.set_ylabel("Length of Q Table")
        #
        # x, y = set_progress_chart(filepath)
        # x = np.concatenate((np.array([0]), x))
        # y = np.concatenate((np.array([0]), y))
        # ax.plot(x, y)
        # ax.xaxis.offsetText.set_fontsize(labelsize)
        # ax.yaxis.offsetText.set_fontsize(labelsize)
        # plt.sca(ax)
        plt.xticks(fontsize=labelsize)
        plt.yticks(fontsize=labelsize)
        # plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    #plt.savefig("data/image_outputs/compare_qlengths.svg", format="svg")
    #plt.show()

        #ax.show()

        plt.title(title)
        set_quantities_histogram(filepath + "quantity-" + obs.get_file_name_string() + ".npy")
        plt.xlabel("Observation frequency")
        plt.ylabel("Count")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

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
    #plt.plot(x, y)

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

    #plt.plot(bins[:-1], n)

if __name__ == '__main__':
    main()

