import datasets
import util
import json
import keras
import sys

import lstm
import cnn
import nn


def experiment_dataset(
    selection_problem,
    selection_source,
    selection_count,
    selection_random_seed,
    selection_reject_minimum,
    selection_overwrite,
    embedding_type,
    embedding_shape,
    embedding_overwrite
):
    # get selection set
    selection_df, name = datasets.get_selection_set(
        problem=selection_problem,
        source=selection_source,
        count=selection_count,
        random_seed=selection_random_seed,
        reject_minimum=selection_reject_minimum,
        overwrite=selection_overwrite,
    )

    # create necessary embedding/vector form
    embedding_df = datasets.get_embedding_set(
        selection_df,
        embedding_type=embedding_type,
        output_name=name,
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )

    return embedding_df, selection_df, name


@util.dump_log
def experiment_model(
    selection_problem,
    selection_source,
    selection_count,
    selection_random_seed,
    selection_reject_minimum,
    selection_overwrite,
    embedding_type,
    embedding_shape,
    embedding_overwrite,
    model_type,
    model_arch_num,
    model_layer_sizes,
    model_maxlen,
    model_batch_size,
    model_learning_rate,
    model_epochs,
    model_num
):
    embed_df, sel_df, name = experiment_dataset(
        selection_problem,
        selection_source,
        selection_count,
        selection_random_seed,
        selection_reject_minimum,
        selection_overwrite,
        embedding_type,
        embedding_shape,
        embedding_overwrite
    )

    X = embed_df
    if selection_problem == "reliability":
        y = sel_df.reliable
    elif selection_problem == "biased" or selection_problem == "extreme_biased":
        y = sel_df.biased

    # pad as needed
    if embedding_shape == "sequence":
        X = lstm.pad_data(X, maxlen=model_maxlen)

    y = keras.utils.to_categorical(y)
        
    if model_type == "lstm":
        model, history = lstm.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs)
    elif model_type == "cnn":
        model, history = cnn.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs)
    elif model_type == "nn":
        model, history = nn.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs)
        pass
    elif model_type == "svm":
        pass

    name = f'{name}_{model_type}_{model_arch_num}_{model_num}_{model_maxlen}_{model_batch_size}_{model_learning_rate}.json'

    with open("../data/output/" + name, "w") as outfile:
        json.dump(history, outfile)


if __name__ == "__main__":
    util.init_logging()
    
    # experiment_model("extreme_biased", "as", 5560, 13, 500, False, "w2v", "sequence", False, "lstm", 2, (64, 32, 2), 500, 32, .001, 100, 1)
    datasets.create_selection_set_sources(10000, 500)
