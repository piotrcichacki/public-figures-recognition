import json

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import sklearn

from utils.data_preprocessing import load_model_input_data, one_hot_encoding, split_data
from utils.utils import load_yaml, plot_confusion_matrix, plot_to_image


def train_model(hparams, version):
    optimizer = hparams[HP_OPTIMIZER]
    units_first = hparams[HP_NUM_UNITS_FIRST]
    units_second = hparams[HP_NUM_UNITS_SECOND]
    lr = hparams[HP_LR]
    parameters_str = (
        "hyperparameters_tunning-optimizer="
        + optimizer
        + ",nr_units_first="
        + str(units_first)
        + ",nr_units_second="
        + str(units_second)
        + ",learning_rate="
        + str(lr)
    )
    model = tf.keras.models.Sequential(name=("Model-MLP-v" + str(version)))
    model.add(
        tf.keras.layers.Dense(
            units=units_first, activation="relu", input_dim=512, name="dense_1"
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.1, name="dropout_in"))
    model.add(
        tf.keras.layers.Dense(units=units_second, activation="relu", name="dense_2")
    )
    model.add(tf.keras.layers.Dropout(rate=0.5, name="dropout_1"))
    model.add(
        tf.keras.layers.Dense(
            units=number_of_people, activation="softmax", name="dense_out"
        )
    )

    print(model.summary())

    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(learning_rate=lr)
            if optimizer == "adam"
            else tf.keras.optimizers.SGD(learning_rate=lr)
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        metrics=["accuracy"],
    )

    log_dir = "logs/model/" + parameters_str
    file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    def log_confusion_matrix(epoch, logs):
        test_pred_raw = model.predict(X_test)
        test_pred = np.argmax(test_pred_raw, axis=1)
        test_target = np.argmax(y_test, axis=1)

        cm = sklearn.metrics.confusion_matrix(test_target, test_pred)

        f = open("data/output/id.json")
        ids = json.load(f)
        classes = [value for _, value in ids.items()]

        figure = plot_confusion_matrix(cm, class_names=classes)
        cm_image = plot_to_image(figure)

        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5)

    callbacks = [tensorboard_callback, cm_callback, early_stopping_callback]

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    results = model.evaluate(X_test, y_test, batch_size=128)
    print(f"Results: test loss = {results[0]:.3f}, test acc = {results[1]:.3f}")
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)
        tf.summary.scalar("accuracy", results[1], step=1)
        tf.summary.scalar("loss", results[0], step=1)


if __name__ == "__main__":

    params = load_yaml("conf/parameters.yml")
    catalog = load_yaml("conf/catalog.yml")

    number_of_people = len(catalog["footballers"])
    train_test_ratio = params["split_sizes"]["train_test"]
    train_val_ratio = params["split_sizes"]["train_val"]

    X, y = load_model_input_data()
    y = one_hot_encoding(y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_test_ratio, train_val_ratio
    )
    print("Full dataset shape: ", X.shape, y.shape)
    print("Training dataset shape: ", X_train.shape, y_train.shape)
    print("Validation dataset shape: ", X_val.shape, y_val.shape)
    print("Testing dataset shape: ", X_test.shape, y_test.shape)

    HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["sgd", "adam"]))
    HP_NUM_UNITS_FIRST = hp.HParam(
        "nr units first dense layer", hp.Discrete([32, 64, 128])
    )
    HP_NUM_UNITS_SECOND = hp.HParam(
        "nr units second dense layer", hp.Discrete([32, 64, 128])
    )
    HP_LR_SGD = hp.HParam("learning rate", hp.Discrete([0.1, 0.01, 1e-3]))
    HP_LR_ADAM = hp.HParam("learning rate", hp.Discrete([1e-3, 1e-4, 1e-5]))

    version = 1

    for optimizer in HP_OPTIMIZER.domain.values:
        for nr_units_first in HP_NUM_UNITS_FIRST.domain.values:
            for nr_units_second in HP_NUM_UNITS_SECOND.domain.values:
                HP_LR = HP_LR_ADAM if optimizer == "adam" else HP_LR_SGD
                for lr in HP_LR.domain.values:
                    hparams = {
                        HP_OPTIMIZER: optimizer,
                        HP_NUM_UNITS_FIRST: nr_units_first,
                        HP_NUM_UNITS_SECOND: nr_units_second,
                        HP_LR: lr,
                    }
                    train_model(hparams, version)
                    version += 1
