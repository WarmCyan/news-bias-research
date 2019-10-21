import keras
from keras.layers import LSTM, Dense, TimeDistributed
from sklearn.model_selection import train_test_split

import util


def create_model(lstm_layer_sizes=[64, 32], fully_connected=[2]):
    model = keras.Sequential()

    for i, layer in enumerate(lstm_layer_sizes):

        # check for last layer
        if i == len(lstm_layer_sizes) - 1:
            model.add(LSTM(layer))
        elif i == 0:
            model.add(LSTM(layer, return_sequences=True, input_shape=(None, 300)))
        else:
            model.add(LSTM(layer, return_sequences=True))

    for layer in fully_connected:
        model.add(
            Dense(layer, activation="sigmoid")
        )  # NOTE: sigmoid as in the "exploration" paper

    return model


def pad_data(data, maxlen=500):
    padded = keras.preprocessing.sequence.pad_sequences(data, maxlen=500)
    return padded


# NOTE: assumes X is already padded and that y is already categorical
def train_test(X, y, lstm_layer_sizes, fully_connected, epochs=1):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.20, random_state=13
    # )

    model = create_model(lstm_layer_sizes, fully_connected)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    model.fit(
        X,
        y,
        batch_size=64,
        verbose=1,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
        workers=4,
    )

    return model


def train(X, y, lstm_layer_sizes, fully_connected):
    pass
