import os
import random
import pickle
import scipy.io.wavfile as wav
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from collections import defaultdict

directory = "data/genres_original/"


def train_model():
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

    # os.system('cls')
    print('Training complete!')
    f.close()
