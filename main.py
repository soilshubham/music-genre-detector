from src.predict import predict
from src.get_accuracy import getAccuracy
from src.get_neighbors import getNeighbors
from src.nearest_class import nearestClass
from src.train import train_model
from src.load_dataset import load_dataset

test_audio = 'data/genres_original/jazz/jazz.00022.wav'


def main():
    # train_model()
    # dataset, training_set, test_set = load_dataset("my.dat", 0.66)
    # leng = len(test_set)
    # predictions = []
    # for x in range(leng):
    #     predictions.append(nearestClass(
    #         getNeighbors(training_set, test_set[x], 10)))

    # accuracy1 = getAccuracy(test_set, predictions)
    # print('Accuracy of model: ', round(accuracy1*100, 2))
    predict(test_audio)


if __name__ == '__main__':
    main()
