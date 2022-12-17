import datetime

import tensorflow as tf

from utils.data_preprocessing import load_model_input_data, one_hot_encoding, split_data
from utils.utils import load_yaml

if __name__ == "__main__":

    log_dir = "logs/model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    params = load_yaml("conf/parameters.yml")
    catalog = load_yaml("conf/catalog.yml")

    number_of_people = len(catalog["footballers"])
    train_test_ratio = params["split_sizes"]["train_test"]
    train_val_ratio = params["split_sizes"]["train_val"]

    X, y = load_model_input_data()
    y = one_hot_encoding(y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_test_ratio, train_val_ratio)
    print("Full dataset shape: ", X.shape, y.shape)
    print("Training dataset shape: ", X_train.shape, y_train.shape)
    print("Validation dataset shape: ", X_val.shape, y_val.shape)
    print("Testing dataset shape: ", X_test.shape, y_test.shape)

    model = tf.keras.models.Sequential(name="Model-MLP-v1")
    model.add(tf.keras.layers.Dense(units=64, activation="relu", input_dim=512, name="dense_1"))
    model.add(tf.keras.layers.Dropout(rate=0.5, name="dropout_1"))
    model.add(tf.keras.layers.Dense(units=32, activation="relu", name="dense_2"))
    model.add(tf.keras.layers.Dense(units=number_of_people, activation="softmax", name="dense_out"))

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5)

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=50,
                        batch_size=128,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard_callback, early_stopping_callback])

    results = model.evaluate(X_test, y_test, batch_size=128)
    print(f"Results: test loss = {results[0]:.3f}, test acc = {results[1]:.3f}")
