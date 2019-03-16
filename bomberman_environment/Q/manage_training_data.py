
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



def get_index(index_file):
    """
    Access the current active length of the Q- and observation tables.
    :param index_file: Filename with index record
    :return:
    """

    try:
        file = open(index_file, 'r')
        index = json.load(file)
        file.close()
        return index
    except:
        print("json file", index_file, "empty, initializing with index = 1.")
        file = open(index_file, 'w')
        json.dump(1, file)
        file.close()
        return 1

def set_index(index_file, new_index_value):

    """
    Set the index value to a new value.
    :param index_file: Filename with index record
    :param new_index_value: New value to update with
    :return:
    """

    f = open(index_file, "w")

    json.dump(new_index_value, f)

    f.close()
