import os
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc

from .load_dataset import load_dataset
from .model import getNeighbors, nearestClass

results = {
    1: 'blues',
    2: 'classical',
    3: 'country',
    4: 'disco',
    5: 'hiphop',
    6: 'jazz'
}


def predict(audioFile: str) -> str:
    """
    Given an audio file, predicts the genre of the audio by extracting
    features using MFCC and using the k-NN algorithm.

    Parameters:
    -----------
    1. audioFile (str): Path to the audio file to be classified (`.wav` format only).

    Returns:
    --------
    str: The predicted genre of the audio file as a string.
    """

    dataset, training_set, test_set = load_dataset(0.66)
    _file = audioFile
    (_rate, _sig) = wav.read(_file)
    _mfcc_feat = mfcc(_sig, _rate, winlen=0.020, appendEnergy=False)
    _covariance = np.cov(np.matrix.transpose(_mfcc_feat))
    _mean_matrix = _mfcc_feat.mean(0)
    _feature = (_mean_matrix, _covariance)

    _pred = nearestClass(getNeighbors(dataset, _feature, 5))
    return results[_pred]


def predict_test(audioFile: str) -> str:
    """
    Given an audio file, predicts the genre of the audio by extracting
    features using MFCC and using the k-NN algorithm.

    `This is only for testing.`

    Parameters:
    -----------
    1. audioFile (str): Path to the audio file to be classified.

    Returns:
    --------
    str: The predicted genre of the audio file as a string.
    """

    dataset, training_set, test_set = load_dataset(0.66)
    _file = os.open(audioFile, os.O_RDWR)
    (_rate, _sig) = wav.read(_file)
    _mfcc_feat = mfcc(_sig, _rate, winlen=0.020, appendEnergy=False)
    _covariance = np.cov(np.matrix.transpose(_mfcc_feat))
    _mean_matrix = _mfcc_feat.mean(0)
    _feature = (_mean_matrix, _covariance)

    _pred = nearestClass(getNeighbors(dataset, _feature, 5))
    print(results[_pred])
