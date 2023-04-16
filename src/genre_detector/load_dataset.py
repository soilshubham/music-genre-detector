import os
from typing import Tuple, List, Any
import pickle
import random


def load_dataset(split: float) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Loads the trained dataset and splits it into training and test sets.

    Parameters:
    -----------
    1. split (float): The fraction of the dataset to use for training.

    Returns:
    --------
    Tuple[List[Any], List[Any], List[Any]]: A tuple of three lists representing the dataset,
    the training set, and the test set, respectively.
    """

    dataset = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'my.dat')

    with open(dataset_path, 'rb') as f:
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
