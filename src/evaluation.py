import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_nearest_neighbor(model, x_test, y_test):
    start = time.perf_counter()
    predictions = model.predict(x_test.reshape(len(x_test), -1)).astype(int)
    elapsed = time.perf_counter() - start
    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "time": elapsed,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(range(10)))
    }

def evaluate_histogram(templates, x_test, y_test):
    from .histogram_classifier import predict_histogram_classifier
    start = time.perf_counter()
    predictions = predict_histogram_classifier(templates, x_test)
    elapsed = time.perf_counter() - start
    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "time": elapsed,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(range(10)))
    }
