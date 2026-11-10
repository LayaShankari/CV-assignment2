"""
src/histogram_classifier.py
---------------------------
Implements an intensity histogram template matching classifier:
1. Calculates a 16-bin normalized grayscale intensity histogram for each image.
2. Averages training histograms per category to construct 10 class templates.
3. Classifies a new sample by computing the Chi-Square distance to each template.
"""

import numpy as np
from .preprocessing import extract_histograms


def train_histogram_classifier(x_train, y_train):
    """
    Build one average normalized intensity histogram for each class (0 through 9).
    """
    train_features = extract_histograms(x_train)
    class_templates = []

    for class_id in range(10):
        class_features = train_features[y_train == class_id]
        if len(class_features) > 0:
            class_templates.append(class_features.mean(axis=0))
        else:
            class_templates.append(np.zeros(16, dtype=np.float32))

    return np.asarray(class_templates, dtype=np.float32)


def predict_histogram_classifier(templates, images):
    """Predict classes for a batch of 14x14 images."""
    test_features = extract_histograms(images)
    predictions = []

    # Chi-square distance: 0.5 * sum((hist_A - hist_B)^2 / (hist_A + hist_B))
    for feature in test_features:
        numerator = (feature[None, :] - templates) ** 2
        denominator = feature[None, :] + templates + 1e-10
        distances = 0.5 * np.sum(numerator / denominator, axis=1)
        predictions.append(int(np.argmin(distances)))

    return np.asarray(predictions, dtype=int)


def predict_one_histogram(templates, image):
    """Predict class for a single 14x14 image."""
    return int(predict_histogram_classifier(templates, image[None, :, :])[0])


def get_histogram_distances(templates, image):
    """
    Returns chi-square distances from the image histogram to all 10 class templates.
    Lower distance = closer match.
    """
    feature = extract_histograms(image[None, :, :])[0]
    numerator = (feature[None, :] - templates) ** 2
    denominator = feature[None, :] + templates + 1e-10
    distances = 0.5 * np.sum(numerator / denominator, axis=1)
    return distances, feature
