import os
import genre_detector as gd
import sys

sys.path.append('/path/to/src/genre_detector')

datasetDir = os.path.join(os.path.dirname(
    __file__), '..', 'data/genres_original/')

if __name__ == '__main__':
    gd.train(datasetDir, accuracy=True)
    # gd.predict_test(datasetDir+'/disco/disco.00032.wav')
