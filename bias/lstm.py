import keras
from keras.layers import LSTM, Dense, Masking
from sklearn.model_selection import train_test_split

import util


def create_model(arch_num, layer_sizes, maxlen):
    model = keras.Sequential()

    model.add(Masking(mask_value=0.0, input_shape=(maxlen, 300)))

    if arch_num == 1:
        model.add(LSTM(layer_sizes[0]))
        model.add(Dense(layer_sizes[1], activation='softmax'))
    elif arch_num == 2:
        model.add(LSTM(layer_sizes[0]))
        model.add(Dense(layer_sizes[1], activation='sigmoid'))
        model.add(Dense(layer_sizes[2], activation='softmax'))
    elif arch_num == 3:
        model.add(LSTM(layer_sizes[0], return_sequences=True))
        model.add(LSTM(layer_sizes[1]))
        model.add(Dense(layer_sizes[2], activation='softmax'))
    elif arch_num == 4:
        model.add(LSTM(layer_sizes[0], return_sequences=True))
        model.add(LSTM(layer_sizes[1]))
        model.add(Dense(layer_sizes[2], activation='sigmoid'))
        model.add(Dense(layer_sizes[3], activation='softmax'))
    elif arch_num == 5:
        model.add(LSTM(layer_sizes[0], return_sequences=True))
        model.add(LSTM(layer_sizes[1]))
        model.add(LSTM(layer_sizes[2]))
        model.add(Dense(layer_sizes[3], activation='sigmoid'))
        model.add(Dense(layer_sizes[4], activation='softmax'))

    return model


def pad_data(data, maxlen=500):
    padded = keras.preprocessing.sequence.pad_sequences(data, maxlen=500, padding='post', dtype='float32')
    return padded


# NOTE: assumes X is already padded and that y is already categorical
@util.dump_log
def train_test(X, y, arch_num, layer_sizes, maxlen, batch_size, learning_rate, epochs=1):
    model = create_model(arch_num, layer_sizes, maxlen)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    model.summary()
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        #verbose=2,
        verbose=1,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
        workers=4,
    )

    return model, history
