from flask import Flask, request, render_template
from genre_detector.predict import predict
from genre_detector.get_accuracy import getAccuracy
from genre_detector.get_neighbors import getNeighbors
from genre_detector.nearest_class import nearestClass
from genre_detector.train import train_model
from genre_detector.load_dataset import load_dataset

test_audio = 'data/genres_original/jazz/jazz.00022.wav'


def main():
    train_model()
    dataset, training_set, test_set = load_dataset("my.dat", 0.66)
    leng = len(test_set)
    predictions = []
    for x in range(leng):
        predictions.append(nearestClass(
            getNeighbors(training_set, test_set[x], 10)))

    accuracy = getAccuracy(test_set, predictions)
    print('Accuracy of model: ', round(accuracy*100, 2), '%')


if __name__ == '__main__':
    main()
