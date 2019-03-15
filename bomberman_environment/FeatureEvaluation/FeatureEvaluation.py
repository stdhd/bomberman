

import numpy as np
from FeatureEvaluation.get_subsets import get_subsets_recursively
from Q.Training import q_train_from_games
from agent_code.observation_object import ObservationObject


class FeatureEvaluation:
    """
    Class to assist with feature testing and comparison.
    """
    def __init__(self, global_output_dir: str, features: list=None, lengths: np.array=np.array([1]),
                 radii: np.array=np.array([5])):
        """
        Initialize a feature evaluation object using a base set of features to choose from and a list of allowable
        "combination lengths", i.e. number of features allowed in a combination. => Set to [1] for only 1 feature, [2, 3]
        for 2 or 3 features, etc.
        :param global_output_dir: Directory into which to output results (each in its own directory)
        :param features: List of features (Important: Names should be in *long* format)
        :param lengths: How many features are allowed at once (array of values)
        :param radii: List of radii to try
        """

        self.global_output_dir = global_output_dir

        self.lengths, self.radii = lengths, radii

        self.training_data_dir, self.results_output_dir = None, None

        _all_combinations = np.array(get_subsets_recursively(features))
        self.combinations = _all_combinations[np.where(len(_all_combinations) in lengths)]


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

    def clear_results_output_dir(self):
        """
        Reset output directory to None
        :return:
        """
        self.results_output_dir = None

    def _run_training(self, radius, feature_combination:np.array):
        """
        Once training data are set, run Q learning a specific radius and feature combination.
        :raise RuntimeError if training directory not initialized
        :return:
        """

        if self.training_data_dir is None:
            raise RuntimeError("Training directory not set")

        obs = ObservationObject(radius, None, feature_combination)

        self.set_results_output_dir(self.global_output_dir + "/" + obs.get_file_name_string())

        q_train_from_games(self.training_data_dir, self.results_output_dir, obs)

        self.clear_results_output_dir()

    def run_all_features(self):
        """
        Master function to train with every allowable combination of features and radius.
        :raise RuntimeError if training directory not initialized
        :return:
        """
        if self.training_data_dir is None:
            raise RuntimeError("Training directory not set")

        print("Starting training suite with", self.combinations.shape[0]*self.radii.shape[0],
              "allowed feature and radius combinations.")
        print("----------")

        for number, features in enumerate(self.combinations):
            print("Training feature set nr.", number, "out of", self.combinations.shape[0])
            print("Features: ", [name for name in features])
            for radius in self.radii:
                print("Starting training for features with radius", radius)
                self._run_training(radius, features)
            print("-----")


    def evaluate_agents(self, agents):
        """
        Launches a tournament using
        :return:
        """