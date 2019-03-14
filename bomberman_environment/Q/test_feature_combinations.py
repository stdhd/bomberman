
import numpy as np

from Q.Training import q_train_from_games_jakob
from agent_code.observation_object import ObservationObject


def main():
    """
    Test all subsets of feature combinations for window sizes in a specified range.
    :return:
    """

    features = np.array((ObservationObject(1, []).name_dict.keys()))

    def get_subsets_recursively(find_subsets):

        if len(find_subsets) == 1:
            return [[], find_subsets.copy()]

        without_last = get_subsets_recursively(find_subsets[:-1].copy())

        with_last = []

        for subset in without_last:
            _subset = subset.copy()
            _subset.append(find_subsets[-1])
            with_last.append(_subset)

        ret = without_last + with_last

        return ret

    feature_combinations = [features[subset] for subset in get_subsets_recursively(list(range(features.shape[0])))]

    #  TODO train here

main()
