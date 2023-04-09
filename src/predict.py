import os
import scipy.io.wavfile as wav
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from collections import defaultdict
from src.get_neighbors import getNeighbors
from src.load_dataset import load_dataset
from src.nearest_class import nearestClass

from collections import defaultdict

results = defaultdict(int)

directory = "data/genres_original"
i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1


def predict(audoFile):
    dataset, training_set, test_set = load_dataset("my.dat", 0.66)
    _file = os.open(audoFile, os.O_RDWR)
    (_rate, _sig) = wav.read(_file)
    _mfcc_feat = mfcc(_sig, _rate, winlen=0.020, appendEnergy=False)
    _covariance = np.cov(np.matrix.transpose(_mfcc_feat))
    _mean_matrix = _mfcc_feat.mean(0)
    _feature = (_mean_matrix, _covariance)

    _pred = nearestClass(getNeighbors(dataset, _feature, 5))
    print(results[_pred])
