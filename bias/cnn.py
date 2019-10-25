import keras
from keras.layers import Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split

import util


# layer size should be [(nodes, kernel), ..., dense_size, dense_size]
def create_model(arch_num, layer_sizes, maxlen):
    model = keras.Sequential()

    if arch_num == 1:
        model.add(Conv1D(layer_sizes[0][0], kernel_size=layer_sizes[0][1], input_shape=(300, maxlen)))
        model.add(Conv1D(layer_sizes[1][0], kernel_size=layer_sizes[1][1]))
        model.add(Flatten())
        model.add(Dense(layer_sizes[2], activation='relu'))
        model.add(Dense(layer_sizes[3], activation='softmax'))

    return model

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
        verbose=2,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
        workers=4,
    )

    return model, history
