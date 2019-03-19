

import pickle
import numpy as np

from sklearn import tree, ensemble, kernel_ridge


def train_decision_tree(KNOWN_filepath:str, QTABLE_filepath:str, writepath:str, cutoff:int = 4):
    """
    Train a classification model using a database of observations and their Q values. Disregard data with less than "cutoff"
     updates.
    :param KNOWN_filepath: Database of observations
    :param QTABLE_filepath: Q Table for observations
    :param writepath Where to dump the model parameters
    :param cutoff Disregard data with fewer updates than this
    :return:
    """

    QTABLE = np.load(QTABLE_filepath)
    KNOWN = np.load(KNOWN_filepath)
    multiplicities = QTABLE[:, 6]

    select = np.where(multiplicities >= cutoff)

    data = KNOWN[select]
    target = np.argmax(QTABLE[:, 0:6][select], axis=1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(data, target)

    pickle.dump(clf, open(writepath, "w"))


def train_decision_forest(KNOWN_filepath:str, QTABLE_filepath:str, writepath:str, cutoff:int = 4):
    """
    Train a classification model using a database of observations and their Q values. Disregard data with less than "cutoff"
     updates.
    :param KNOWN_filepath: Database of observations
    :param QTABLE_filepath: Q Table for observations
    :param writepath Where to dump the model parameters
    :param cutoff Disregard data with fewer updates than this
    :return:
    """

    QTABLE = np.load(QTABLE_filepath)
    KNOWN = np.load(KNOWN_filepath)
    multiplicities = QTABLE[:, 6]

    select = np.where(multiplicities >= cutoff)

    data = KNOWN[select]
    target = np.argmax(QTABLE[:, 0:6][select], axis=1)

    clf = ensemble.RandomForestClassifier()
    clf.fit(data, target)

    pickle.dump(clf, open(writepath, "w"))


def train_kernel_regression(KNOWN_filepath:str, QTABLE_filepath:str, writepath:str, cutoff:int = 4):
    """
    Train a regression model using a database of observations and their Q values. Disregard data with less than "cutoff"
     updates.
    :param KNOWN_filepath: Database of observations
    :param QTABLE_filepath: Q Table for observations
    :param writepath Where to dump the model parameters
    :param cutoff Disregard data with fewer updates than this
    :return:
    """

    QTABLE = np.load(QTABLE_filepath)
    KNOWN = np.load(KNOWN_filepath)
    multiplicities = QTABLE[:, 6]

    select = np.where(multiplicities >= cutoff)[0]

    data = np.tile(KNOWN[select], (6, 1))

    target = (QTABLE[:, 0:6][select]).T.flatten()

    clf = kernel_ridge.KernelRidge(kernel="rbf")
    clf.fit(data, target)

    pickle.dump(clf, open(writepath, "w"))
