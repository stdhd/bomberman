

import numpy as np

from Q.manage_training_data import Table

def main():
    """
    Test basic loading/updating capabilities of Table object.
    :return:
    """

    filepath = 'data/qtables/TEST'

    arr_fp = filepath+"/test.npy"

    np.save(arr_fp, np.zeros((30, 20)))

    table_wrapper = Table(filepath+"/qtable.json")

    table_wrapper.set_table(np.load(arr_fp))

    indices = np.array([np.random.choice(np.arange(30), replace=False) for i in range(10)])

    vals = np.random.rand((10, 20))

    table_wrapper.insert_into_table(vals, indices)



