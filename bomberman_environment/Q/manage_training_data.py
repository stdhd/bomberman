
import json


def check_if_trained(records_file, train_data_file):
    """
    Check a records file if a file containing training data has already been used for training.

    Records file should be in JSON format.

    :param records_file: File containing list of training files already used for training.
    :param train_data_file: File containing training data
    :raises IO Exceptions
    :return: True iff training data file has already been used for training.
    """

    with open(records_file, "r") as f:
        records = json.load(f)

    if train_data_file in records:
        f.close()
        return True
    return False


def add_to_trained(records_file, train_data_file):
    """
    Add a file used for training to records file.
    :param records_file: File containing list of training files already used for training.
    :param train_data_file: File containing training data.
    :return: True iff file loading successful
    """

    with open(records_file, "w") as f:
        records = json.load(f)

    records.append(train_data_file)

    json.dump(records, f)

    f.close()





