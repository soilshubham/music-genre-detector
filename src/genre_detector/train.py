import os
import pickle
from typing import Optional
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc

from .model import getAccuracy

def train(directory: str, accuracy: Optional[bool] = False) -> None:
    """
    Trains a K-Nearest Neighbors (K-NN) model for genre recognition and saves it in a binary file 'my.dat'.
    
    Parameters:
    -----------
    1. directory (str): A string representing the path to the directory containing the audio files dataset.
    2. accuracy(bool, optional) : A flag indicating whether to calculate and print model accuracy.
    
    Returns:
    --------
    None
    """

    f = open("my.dat", 'wb')
    i = 0

    print('training...')
    for folder in os.listdir(directory):
        i += 1
        if i == 11:
            break
        for file in os.listdir(directory+folder):
            (rate, sig) = wav.read(directory+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)

    print('Training complete!')

    # Getting accuracy
    if accuracy:
        accuracy = getAccuracy()
        print('Model accuracy: ', round(accuracy*100, 2), '%')

    f.close()
