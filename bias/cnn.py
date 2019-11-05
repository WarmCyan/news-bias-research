import keras
from keras import callbacks
from keras.layers import Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split
import logging

import util


# layer size should be [(nodes, kernel), ..., dense_size, dense_size]
def create_model(arch_num, layer_sizes, maxlen):
    model = keras.Sequential()

    if arch_num == 1:
        model.add(Conv1D(layer_sizes[0][0], kernel_size=layer_sizes[0][1], input_shape=(300*500, maxlen)))
        model.add(Conv1D(layer_sizes[1][0], kernel_size=layer_sizes[1][1]))
        model.add(Flatten())
        model.add(Dense(layer_sizes[2], activation='relu'))
        model.add(Dense(layer_sizes[3], activation='softmax'))

    return model

# NOTE: assumes X is already padded and that y is already categorical
@util.dump_log
def train_test(X, y, arch_num, layer_sizes, maxlen, batch_size, learning_rate, epochs, X_test, y_test, name):
    model = create_model(arch_num, layer_sizes, maxlen)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]#, callbacks.ModelCheckpoint(filepath='../models/' + name + '.best.weights', verbose=1, save_best_only=True)]
    
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    model.summary()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, stratify=y, test_size=.2, random_state=13)
    
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        verbose=2,
        epochs=epochs,
        validation_data=(X_val, y_val),
        # validation_split=0.2,
        # use_multiprocessing=True,
        # workers=4,
        callbacks=cbks
    )
    
    #model.load_weights("../models/" + name + ".best.weights")
    
    loss, acc, predictions = test(X_test, y_test, batch_size, model)

    return model, history.history, loss, acc


def test(X, y, batch_size, model):
    predictions = model.predict(X, batch_size)
    loss, acc = model.evaluate(X, y, batch_size, show_accuracy=True)
    logging.info('Test loss / test accuracy: %f / %f', loss, acc)
    return loss, acc, predictions
