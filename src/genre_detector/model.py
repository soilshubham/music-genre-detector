from typing import List, Tuple
import numpy as np
import operator
from .load_dataset import load_dataset


def distance(instance1: Tuple[np.ndarray, np.ndarray], instance2: Tuple[np.ndarray, np.ndarray], k: int) -> float:
    """
    Calculates the Mahalanobis distance between two instances.

    Parameters:
    -----------
    1. instance1 (Tuple): A tuple containing the mean vector and covariance matrix
    of the first instance.
    2. instance2 (Tuple): A tuple containing the mean vector and covariance matrix
    of the second instance.
    3. k (int): A constant value to subtract from the distance.

    Returns:
    --------
    float: The Mahalanobis distance between instance1 and instance2, adjusted by k.
    """

    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(),
                 np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbors(trainingSet: List[Tuple[np.ndarray, np.ndarray, str]], instance: Tuple[np.ndarray, np.ndarray, str], k: int) -> List[str]:
    """
    Returns the k-nearest neighbors of an instance in a training set, based on the Mahalanobis distance.

    Parameters:
    -----------
    1. trainingSet (List): A list of tuples, where each tuple contains
    the mean vector, covariance matrix, and label of a training instance.
    2. instance (Tuple): A tuple containing the mean vector, covariance matrix,
    and label of the instance for which to find the k-nearest neighbors.
    3. k (int): The number of nearest neighbors to return.

    Returns:
    --------
    List[str]: A list of labels of the k-nearest neighbors to the input instance.
    """

    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + \
            distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestClass(neighbors: List[str]) -> str:
    """
    Returns the class label that appears most frequently in a list of nearest neighbors.

    Parameters:
    -----------
    1. neighbors (List[str]): A list of class labels of nearest neighbors.

    Returns:
    --------
    str: The class label that appears most frequently in the input list.
    """

    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(),
                    key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def getAccuracy() -> float:
    """
    Returns the classification accuracy of the K-nearest neighbors algorithm on a test dataset.

    Parameters:
    -----------
    None

    Returns:
    --------
    float: The classification accuracy, defined as the percentage of test instances 
    that are correctly classified by the algorithm.
    """

    dataset, training_set, test_set = load_dataset(0.66)
    leng = len(test_set)
    predictions = []

    for x in range(leng):
        predictions.append(nearestClass(
            getNeighbors(training_set, test_set[x], 10)))

    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(test_set)
