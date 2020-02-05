import keras
from keras import callbacks
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import logging

import util


def create_model(arch_num, layer_sizes, maxlen, data_width, selection_problem):
    model = keras.Sequential()
    
    if arch_num == 1:
        model.add(Dense(layer_sizes[0], activation='relu', input_shape=(data_width,)))
        #model.add(Dense(layer_sizes[1], activation='sigmoid'))
    if arch_num == 2:
        model.add(Dense(layer_sizes[0], activation='relu', input_shape=(data_width,)))
        model.add(Dense(layer_sizes[1], activation='relu'))
        #model.add(Dense(layer_sizes[2], activation='sigmoid'))
    if arch_num == 3:
        model.add(Dense(layer_sizes[0], activation='relu', input_shape=(data_width,)))
        model.add(Dense(layer_sizes[1], activation='relu'))
        model.add(Dense(layer_sizes[2], activation='relu'))
        #model.add(Dense(layer_sizes[3], activation='sigmoid'))

    if selection_problem == "bias_direction":
        model.add(Dense(layer_sizes[-1], activation='softmax'))
    else:
        model.add(Dense(layer_sizes[-1], activation='sigmoid'))
    return model

# TODO: unclear if data width needed here or not, data should always have same meta shape
@util.dump_log
def train_test(X, y, arch_num, layer_sizes, maxlen, batch_size, learning_rate, epochs, X_test, y_test, name, data_width, selection_problem):
    model = create_model(arch_num, layer_sizes, maxlen, data_width, selection_problem)
    print("DATA WIDTH:", data_width)

    weight_file = "../models/" + name + ".weights"
    
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    #tensorboard = callbacks.TensorBoard(log_dir="../logs",
    #                      histogram_freq=1, 
    #                      batch_size=batch_size, 
    #                      write_graph=True, 
    #                      write_grads=True, 
    #                      write_images=False, 
    #                      embeddings_freq=0, 
    #                      embeddings_layer_names=None, 
    #                      embeddings_metadata=None) 

    cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=10), callbacks.ModelCheckpoint(filepath=util.TMP_PATH + name + '.best.weights', verbose=0, save_best_only=True)]
    
    if layer_sizes[-1] == 1:
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"]
        )
    else:
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
        )
    model.summary()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, stratify=y, test_size=.2, random_state=13)

            
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        verbose=2,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=cbks
    )
    
    model.load_weights(util.TMP_PATH + name + ".best.weights")
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
