"""
This package provides tools for genre detection from audio files.

Functions:
- train(accuracy=False) -> None:
    Trains a model using audio files in a specified directory and saves the resulting dataset to a binary file.
    If the `accuracy` parameter is True, also prints the model accuracy.

- predict(audioFile) -> str:
    Predicts the genre of an audio file using a trained model and returns the predicted genre as a string.


Usage example:

    >>> import genre_detection
    >>> genre_detection.train(accuracy=True)
    training...
    Training complete!
    Model accuracy: 86.67%
    >>> genre_detection.predict("myaudio.wav")
    'Rock'
"""

from .train import train
from .predict import predict, predict_test
