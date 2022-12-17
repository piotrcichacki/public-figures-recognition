import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_model_input_data():
    X = np.load("data/05_model/X.npy")
    y = np.load("data/05_model/y.npy")

    return X, y


def one_hot_encoding(y):
    return tf.keras.utils.to_categorical(y)


def split_data(X, y, train_test_ratio, train_val_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_val_ratio, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
