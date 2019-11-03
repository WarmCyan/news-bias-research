import keras
from keras imort callbacks
from keras.layers import LSTM, Dense, Masking, Dropout
from sklearn.model_selection import train_test_split

import util


def create_model(arch_num, layer_sizes, maxlen):
    model = keras.Sequential()

    model.add(Masking(mask_value=0.0, input_shape=(maxlen, 300)))

    if arch_num == 1:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[1], activation='sigmoid'))
    elif arch_num == 2:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[1], activation='tanh'))
        model.add(Dropout(.2))
        model.add(Dense(layer_sizes[2], activation='sigmoid'))
    elif arch_num == 3:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2))
        model.add(LSTM(layer_sizes[1], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[2], activation='sigmoid'))
    elif arch_num == 4:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2))
        model.add(LSTM(layer_sizes[1], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[2], activation='tanh'))
        model.add(Dropout(.2))
        model.add(Dense(layer_sizes[3], activation='sigmoid'))
    # elif arch_num == 2:
    #     model.add(LSTM(layer_sizes[0]))
    #     model.add(Dense(layer_sizes[1], activation='sigmoid'))
    #     model.add(Dense(layer_sizes[2], activation='sigmoid'))
    # elif arch_num == 3:
    #     model.add(LSTM(layer_sizes[0], return_sequences=True))
    #     model.add(LSTM(layer_sizes[1]))
    #     model.add(Dense(layer_sizes[2], activation='sigmoid'))
    # elif arch_num == 4:
    #     model.add(LSTM(layer_sizes[0], return_sequences=True))
    #     model.add(LSTM(layer_sizes[1]))
    #     model.add(Dense(layer_sizes[2], activation='sigmoid'))
    #     model.add(Dense(layer_sizes[3], activation='sigmoid'))
    # elif arch_num == 5:
    #     model.add(LSTM(layer_sizes[0], return_sequences=True))
    #     model.add(LSTM(layer_sizes[1]))
    #     model.add(LSTM(layer_sizes[2]))
    #     model.add(Dense(layer_sizes[3], activation='sigmoid'))
    #     model.add(Dense(layer_sizes[4], activation='sigmoid'))

    return model


def pad_data(data, maxlen=500):
    padded = keras.preprocessing.sequence.pad_sequences(data, maxlen=500, padding='post', dtype='float32')
    return padded


# NOTE: assumes X is already padded and that y is already categorical
@util.dump_log
def train_test(X, y, arch_num, layer_sizes, maxlen, batch_size, learning_rate, epochs=1, X_test, y_test):
    model = create_model(arch_num, layer_sizes, maxlen)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # fname = 'weights/keras-lstm.h5'
    # model.load_weights(fname)
    # cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
    #         callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=5)]

    if layer_sizes[-1] == 1:
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"]
        )
    else:
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
        )
    model.summary()
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        verbose=2,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
        workers=4,
        callbacks=cbks
    )

    loss, acc = test(X_test, y_test, batch_size, model)

    return model, history, loss, acc


def test(X, y, batch_size, model):
    loss, acc = model.evaluate(X, y, batch_size, show_accuracy=True)
    logging.info('Test loss / test accuracy: %f / %f', loss, acc)
    return loss, acc
