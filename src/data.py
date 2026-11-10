import os
import gzip
import numpy as np
import streamlit as st
from .preprocessing import preprocess_array

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_raw_fashion_mnist():
    """
    Loads Fashion-MNIST dataset from local cached files or falls back to Keras downloader.
    Ensures offline reliability and fast startup.
    """
    # 1. Check local project data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_data_dir = os.path.join(base_dir, "data", "fashion-mnist")
    keras_data_dir = os.path.expanduser("~/.keras/datasets/fashion-mnist")

    data_dir = None
    if os.path.exists(local_data_dir) and os.path.exists(os.path.join(local_data_dir, "train-images-idx3-ubyte.gz")):
        data_dir = local_data_dir
    elif os.path.exists(keras_data_dir) and os.path.exists(os.path.join(keras_data_dir, "train-images-idx3-ubyte.gz")):
        data_dir = keras_data_dir

    if data_dir is not None:
        with gzip.open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), "rb") as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "rb") as imgpath:
            x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), "rb") as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as imgpath:
            x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
        return (x_train, y_train), (x_test, y_test)

    # Fallback to Keras downloader if local files not found
    from tensorflow.keras.datasets import fashion_mnist
    return fashion_mnist.load_data()


@st.cache_data(show_spinner="Loading and preparing Fashion-MNIST...")
def load_fashion_mnist(train_count=10000, test_count=2000):
    (x_train, y_train), (x_test, y_test) = load_raw_fashion_mnist()

    # Random, repeatable subsets keep the demo fast on ordinary laptops.
    rng = np.random.default_rng(42)
    train_indices = rng.choice(len(x_train), min(train_count, len(x_train)), replace=False)
    test_indices = rng.choice(len(x_test), min(test_count, len(x_test)), replace=False)

    x_train_sub = preprocess_array(x_train[train_indices])
    y_train_sub = y_train[train_indices].astype(int)
    x_test_sub = preprocess_array(x_test[test_indices])
    y_test_sub = y_test[test_indices].astype(int)

    return x_train_sub, y_train_sub, x_test_sub, y_test_sub
