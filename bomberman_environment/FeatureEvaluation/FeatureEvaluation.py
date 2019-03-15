

import numpy as np
from FeatureEvaluation.get_subsets import get_subsets_recursively
from Q.Training import q_train_from_games


class FeatureEvaluation:
    """
    Class to assist with feature testing and comparison.
    """
    def __init__(self, features: np.array=np.zeros(0), lengths: np.array=np.array([1])):
        """
        Initialize a feature evaluation object using a base set of features to choose from and a list of allowable
        "combination lengths", i.e. number of features allowed in a combination. => Set to [1] for only 1 feature, [2, 3]
        for 2 or 3 features, etc.
        :param features: List of features (Important: Names should be in *long* format)
        :param lengths:
        """

        self.features, self.lengths = features, lengths

        self.training_data_dir, self.results_output_dir = None, None

        _all_combinations = get_subsets_recursively(features)
        self.combinations = _all_combinations[np.where(_all_combinations.shape[0] in lengths)]


    def set_training_data_dir(self, training_dir: str):
        """
        Set a training directory from which to source games.
        Games must be in a format usable by q training function.
        :param training_dir:
        :return:
        """
        self.training_data_dir = training_dir

    def set_results_output_dir(self, results_dir: str):
        """
        Set the results output directory for a single training cycle (save qtables and json records here)
        :param results_dir:
        :return:
        """
        self.results_output_dir = results_dir


    def run_training(self, feature_combination:np.array):
        """
        Once training data and output locations are set, run Q learning with a certain feature combination.
        :raise RuntimeError if directories not initialized
        :return:
        """

        if self.training_data_dir is None or self.results_output_dir is None:
            raise RuntimeError("Training/Output directories not set")

        q_train_from_games(self.training_data_dir, self.results_output_dir)