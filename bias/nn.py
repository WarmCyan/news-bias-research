import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import util


def create_model(arch_num, layer_sizes, maxlen):
    model = keras.Sequential()
    
    if arch_num == 1:
        model.add(Dense(layer_sizes[0], activation='relu'))
        model.add(Dense(layer_sizes[1], activation='softmax'))
    if arch_num == 2:
        model.add(Dense(layer_sizes[0], activation='relu'))
        model.add(Dense(layer_sizes[1], activation='relu'))
        model.add(Dense(layer_sizes[2], activation='softmax'))
    if arch_num == 3:
        model.add(Dense(layer_sizes[0], activation='relu'))
        model.add(Dense(layer_sizes[1], activation='relu'))
        model.add(Dense(layer_sizes[2], activation='relu'))
        model.add(Dense(layer_sizes[3], activation='softmax'))

    return model

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
        verbose=2,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
        workers=4,
    )

    return model, history
