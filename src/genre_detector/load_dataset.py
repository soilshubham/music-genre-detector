from typing import Tuple, List, Any
import pickle
import random


def load_dataset(filePath: str, split: float) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Loads a dataset from a file and splits it into training and test sets.

    Parameters:
    -----------
    1. filePath (str): The path of the file containing the dataset.
    2. split (float): The fraction of the dataset to use for training.

    Returns:
    --------
    Tuple[List[Any], List[Any], List[Any]]: A tuple of three lists representing the dataset,
    the training set, and the test set, respectively.
    """

    dataset = []

    with open(filePath, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    training_set = []
    test_set = []

    for x in range(len(dataset)):
        if random.random() < split:
            training_set.append(dataset[x])
        else:
            test_set.append(dataset[x])

    return dataset, training_set, test_set
