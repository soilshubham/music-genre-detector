import os
import pickle
import random
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np


def load_dataset(filename, split):
    dataset = []
    with open(filename, 'rb') as f:
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
