

import pickle
import numpy as np
import os

from sklearn import tree, ensemble, kernel_ridge
from agent_code.observation_object import ObservationObject

def train_decision_tree(KNOWN_filepath:str, QTABLE_filepath:str, QUANTITIES_filepath:str, writepath:str, cutoff:int = 40):
    """
    Train a classification model using a database of observations and their Q values. Disregard data with less than "cutoff"
     updates.
    :param KNOWN_filepath: Database of observations
    :param QTABLE_filepath: Q Table for observations
    :param QUANTITIES_filepath: Counts updates in Q Table
    :param writepath Where to dump the model parameters
    :param cutoff Disregard data with fewer updates than this
    :return:
    """

    QTABLE = np.load(QTABLE_filepath)
    KNOWN = np.load(KNOWN_filepath)
    QUANTITIES = np.load(QUANTITIES_filepath)

    select = (QUANTITIES >= cutoff)
    rows_data_indices = np.where(select.all(axis=1))

    data = KNOWN[rows_data_indices]
    target = np.argmax(QTABLE[rows_data_indices], axis=1)
    clf = tree.DecisionTreeClassifier()
    clf.fit(data, target)

    pickle.dump(clf, open(writepath, "wb"))

    return clf


def train_decision_forest(KNOWN_filepath:str, QTABLE_filepath:str, QUANTITIES_filepath:str, writepath:str, cutoff:int = 4):
    """
    Train a classification model using a database of observations and their Q values. Disregard data with less than "cutoff"
     updates.
    :param KNOWN_filepath: Database of observations
    :param QTABLE_filepath: Q Table for observations
    :param QUANTITIES_filepath: Counts updates in Q Table
    :param writepath Where to dump the model parameters
    :param cutoff Disregard data with fewer updates than this
    :return:
    """

    QTABLE = np.load(QTABLE_filepath)
    KNOWN = np.load(KNOWN_filepath)
    QUANTITIES = np.load(QUANTITIES_filepath)

    select = (QUANTITIES >= cutoff)
    rows_data_indices = np.where(select.all(axis=1))

    data = KNOWN[rows_data_indices]
    target = np.argmax(QTABLE[rows_data_indices], axis=1)

    clf = ensemble.RandomForestClassifier()
    clf.fit(data, target)

    pickle.dump(clf, open(writepath, "wb"))

    return clf


def train_kernel_regression(KNOWN_filepath:str, QTABLE_filepath:str, QUANTITIES_filepath:str, writepath:str, cutoff:int = 4):
    """
    Train a regression model using a database of observations and their Q values. Disregard data with less than "cutoff"
     updates.
    :param KNOWN_filepath: Database of observations
    :param QTABLE_filepath: Q Table for observations
    :param QUANTITIES_filepath: Counts updates in Q Table
    :param writepath Where to dump the model parameters
    :param cutoff Disregard data with fewer updates than this
    :return:
    """

    QTABLE = np.load(QTABLE_filepath)
    KNOWN = np.load(KNOWN_filepath)
    QUANTITIES = np.load(QUANTITIES_filepath)

    KNOWN_extended = np.tile(KNOWN, (6, 1))

    actions = []

    for i in range(6):
        actions.append(np.ones(KNOWN.shape[0])*i)

    action_codes = np.concatenate(tuple(actions))
    qtable_cols = np.concatenate(tuple([QTABLE[:, i] for i in range(6)]))
    quantities_cols = np.concatenate(tuple([QUANTITIES[:, i] for i in range(6)]))

    KNOWN_extended = np.append(KNOWN_extended, action_codes[:, np.newaxis], axis=1)
    target_rows = np.where(quantities_cols >= cutoff)

    data = KNOWN_extended[target_rows]
    target = qtable_cols[target_rows]

    clf = kernel_ridge.KernelRidge(kernel="poly")
    clf.fit(data, target)

    pickle.dump(clf, open(writepath, "wb"))

    return clf

def main():
    """
    Train and dump regression models from data
    :return:
    """

    obs0 = ObservationObject(0, ['d_closest_coin_dir',
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

    os.chdir(os.path.dirname(__file__))
    cwd = os.getcwd()

    path = cwd + "/data/qtables/"+obs0.get_file_name_string() + "/"

    name = obs0.get_file_name_string()#"r1_ismal_ismbr_ismcu_ismdd_bbdd_ccdir_ced_csfdir_ded_mhb"

    obspath = path + "observation-" + name + ".npy"
    qpath = path + "q_table-" + name + ".npy"
    quantpath = path + "quantity-" + name + ".npy"

    outdir = "data/regression_results"

    cutoff = 20

    # params = (obspath, qpath, quantpath, outdir, cutoff)

    dt = train_decision_tree(obspath, qpath, quantpath, outdir + "/2019-03-19 17-12-25_2258.dt.p", cutoff)
    #df = train_decision_forest(obspath, qpath, quantpath, outdir + "/df.p", cutoff)
    kr = train_kernel_regression(obspath, qpath, quantpath, outdir + "/2019-03-19 17-12-25_2258.kr.p", cutoff)

    print()

main()


#2019-03-19 17-12-25_2258