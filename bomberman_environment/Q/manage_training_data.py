
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
        pass
        if train_data_file in records:
            return True
        return False
    except json.decoder.JSONDecodeError:
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





