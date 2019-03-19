
import numpy as np
import json


def is_trained(records_file, train_data_file):
    """
    Check a records file if a file containing training data has already been used for training.

    Creates file if it does not exist already.

    Records file should be in JSON format.

    :param records_file: File containing list of training files already used for training.
    :param train_data_file: File containing training data
    :raises IO Exceptions
    :return: True iff training data file has already been used for training.
    """

    try:
        file = open(records_file, 'r')
        records = json.load(file)
        if train_data_file in records:
            return True
        return False
    except:
        print("json file", records_file, "empty, initializing with empty list.")
        file = open(records_file, 'w')
        json.dump([], file)
        file.close()
        return False



def add_to_trained(records_file, train_data_file):
    """
    Add a file used for training to records file.
    Assumes records file exists and contains a list.
    :param records_file: File containing list of training files already used for training.
    :param train_data_file: File containing training data.
    :return: True iff file loading successful
    """

    with open(records_file, "r") as f:
        records = json.load(f)

    records.append(train_data_file)

    f = open(records_file, "w")

    json.dump(records, f)

    f.close()


class Table:
    """
    Provide functions to access, maintain, and grow Q table (basically a dynamic list, but as a numpy array)
    """

    def __init__(self, TABLE:np.array = None):
        """
        Initialize a Table object from table position index in file.
        :param from_existing True if the
        """
        self.TABLE = None
        self._TABLE_POSITION = None  # how many elements are currently stored in the table

        if TABLE is not None:
            self.set_table(TABLE)

        self.growth_factor = 4  # used for growing table size, should be int >= 2

    def set_table(self, TABLE:np.array):
        """
        Set the data for the Table object (this should be a "totally normal" numpy array which you want to store)
        :param TABLE:
        :return:
        """
        self.TABLE = TABLE

        self._TABLE_POSITION = TABLE.shape[0]


    def get_table(self):
        """
        Return the table's data.
        :return:
        """
        return self.TABLE[0:self._TABLE_POSITION]

    def _grow_table(self):
        """
        Multiply the current capacity of the table by a constant factor (called when TABLE_INDEX == _TABLE_CAPACITY)
        :return:
        """
        temp = self.TABLE
        update = np.zeros((temp.shape[0]*self.growth_factor, temp.shape[1]))  # increase the length
        update[0:temp.shape[0]] = temp  # rewrite old values
        self.TABLE = update

    def insert_into_table(self, values: np.array, indices: np.array):

        """
        Insert values[i] into table at position indices[i] in an efficient way.

        WARNING: Assumes TABLE is long enough to support all insertions!
        :param TABLE: Table into which to insert values
        :param values: Values to insert
        :param indices: Where to insert the values
        :return:
        """


        #
        # sort = np.argsort(indices)
        # values = values[sort]
        #
        # indices = indices.astype(int)[sort]
        #
        # concat = [None for i in range(indices.shape[0]*2 + 1)]
        # concat[0] = self.TABLE[0:indices[0]]
        #
        # for num, index in enumerate(indices):
        #
        #     concat[1 + 2*num] = values[num]
        #
        #     if num == indices.shape[0] - 1:
        #         concat[-1] = self.TABLE[index:self._TABLE_POSITION]
        #         continue
        #
        #     concat[2 + 2*num] = self.TABLE[index : indices[num+1]]
        #
        # insert_new = np.concatenate(tuple(concat), out=self.TABLE[0:self._TABLE_POSITION + indices.shape[0]])


        if not values.shape[0] == indices.shape[0]:
            raise RuntimeError("Number of indices and number of values to insert do not match")

        while self.TABLE.shape[0] < self._TABLE_POSITION + indices.shape[0]:
            self._grow_table()

        data = self.TABLE[0:self._TABLE_POSITION]

        np.insert(data, )

        self._TABLE_POSITION += indices.shape[0]


    def get_index(self) -> int:
        """

        Get the table index from a file.
        Access the current active length of the Q- and observation tables.
        :param index_file: Filename with index record
        :return:
        """

        index_file = self.index_file

        try:
            file = open(index_file, 'r')
            index = int(json.load(file))
            file.close()
            return index
        except:
            print("json file", index_file, "empty, initializing with index = TABLE.shape[0].")
            file = open(index_file, 'w')
            json.dump(self._TABLE_POSITION, file)
            file.close()
            return self._TABLE_POSITION

    def set_index(self):
        """
        Set the index value to a new value using object members.
        :return:
        """

        f = open(self.index_file, "w")

        json.dump(int(self._TABLE_POSITION), f)

        f.close()


