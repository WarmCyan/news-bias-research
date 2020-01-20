import keras
from keras import callbacks
from keras.layers import LSTM, Dense, Masking, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import logging
import traceback

import util


def create_model(arch_num, layer_sizes, maxlen, data_width):
    model = keras.Sequential()

    model.add(Masking(mask_value=0.0, input_shape=(maxlen, data_width)))

    if arch_num == 1:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[1], activation='sigmoid'))
    elif arch_num == 2:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[1], activation='tanh'))
        model.add(Dropout(.2))
        model.add(Dense(layer_sizes[2], activation='sigmoid'))
    elif arch_num == 3:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2, return_sequences=True))
        model.add(LSTM(layer_sizes[1], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[2], activation='sigmoid'))
    elif arch_num == 4:
        model.add(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2, return_sequences=True))
        model.add(LSTM(layer_sizes[1], dropout=.2, recurrent_dropout=.2))
        model.add(Dense(layer_sizes[2], activation='tanh'))
        model.add(Dropout(.2))
        model.add(Dense(layer_sizes[3], activation='sigmoid'))
    elif arch_num == 5:
        model.add(Bidirectional(LSTM(layer_sizes[0], dropout=.2, recurrent_dropout=.2)))
        model.add(Dense(layer_sizes[1], activation='sigmoid'))
        

        
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
    padded = keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding='post', dtype='float32')
    return padded


# NOTE: assumes X is already padded and that y is already categorical
@util.dump_log
def train_test(X, y, arch_num, layer_sizes, maxlen, batch_size, learning_rate, epochs, X_test, y_test, name, data_width):
    model = create_model(arch_num, layer_sizes, maxlen, data_width)
    logging.debug("Model created")

    weight_file = "../models/" + name + ".weights"
    
    # optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam(lr=learning_rate, clipnorm=1.0)
    logging.debug("Optimizer ready")

    # fname = 'weights/keras-lstm.h5'
    # model.load_weights(fname)
    # cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
    #         callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=10), callbacks.ModelCheckpoint(filepath=util.TMP_PATH + name + '.best.weights', verbose=0, save_best_only=True)]

    logging.debug("About to compile")
    if layer_sizes[-1] == 1:
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"]
        )
    else:
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
        )
    model.summary()


    return_history = None
    if not util.check_output_necessary(weight_file, True):
        model.load_weights(weight_file)
    else:
        logging.debug("About to split")
        X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, stratify=y, test_size=.2, random_state=13)

        
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            validation_data=(X_val, y_val),
            # validation_split=0.2,
            # use_multiprocessing=True,
            # workers=4,
            callbacks=cbks
        )

        model.load_weights(util.TMP_PATH + name + ".best.weights")
        model.save_weights(weight_file)

        return_history = history.history

    logging.debug("Testing")
    loss, acc, predictions = test(X_test, y_test, batch_size, model)

    logging.debug("Returning...")
    return model, return_history, loss, acc, predictions


def test(X, y, batch_size, model):
    predictions = model.predict(X, batch_size)
    loss, acc = model.evaluate(X, y, batch_size)
    logging.info('Test loss / test accuracy: %f / %f', loss, acc)
    return loss, acc, predictions
