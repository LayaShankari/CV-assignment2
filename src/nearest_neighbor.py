"""
src/nearest_neighbor.py
-----------------------
Implements 1-Nearest Neighbor (1-NN) classification using Euclidean distance
on flattened 14x14 (196-dimensional) image vectors.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def train_nearest_neighbor(x_train, y_train):
    """
    Train a 1-Nearest Neighbor classifier on flattened image pixels.
    Uses Euclidean distance metric.
    """
    model = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    flat_train = x_train.reshape(len(x_train), -1)
    model.fit(flat_train, y_train)
    return model


def predict_nearest_neighbor(model, images):
    """Predict classes for a batch of images."""
    flat_images = images.reshape(len(images), -1)
    return model.predict(flat_images).astype(int)


def predict_one_nearest_neighbor(model, image):
    """Predict class for a single 14x14 image."""
    flat_image = image.reshape(1, -1)
    return int(model.predict(flat_image)[0])


def find_closest_training_example(model, image, x_train, y_train):
    """
    Finds the exact closest training image and distance to explain
    WHY the 1-NN classifier made its prediction.

    Returns:
        tuple: (predicted_class, distance, closest_train_image, closest_train_label)
    """
    flat_image = image.reshape(1, -1)
    distances, indices = model.kneighbors(flat_image, n_neighbors=1, return_distance=True)
    idx = int(indices[0][0])
    dist = float(distances[0][0])
    predicted_class = int(y_train[idx])
    closest_image = x_train[idx]
    return predicted_class, dist, closest_image, predicted_class
