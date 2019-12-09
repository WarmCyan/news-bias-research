import datasets
import pickle
import logging
import argparse
import util
import json
import keras
import sys
import pandas as pd
import numpy as np

import lstm
import cnn
import nn



def experiment_dataset_bias(
    selection_problem,
    selection_test_fold,
    selection_count,
    selection_random_seed,
    selection_overwrite,
    embedding_type,
    embedding_shape,
    embedding_overwrite,
):
    #selection_df, name = 

    name = "{0}_{1}".format(problem, random_seed) 

    selection_df, selection_test_df = datasets.load_folds(selection_test_fold, selection_count, True, selection_overwrite)

    embedding_df = datasets.get_embedding_set(
        selection_df,
        embedding_type=embedding_type,
        output_name=name,
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )
    
    embedding_test_df = datasets.get_embedding_set(
        selection_test_df,
        embedding_type=embedding_type,
        output_name=name,
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )

    return embedding_df, selection_df, name + "fold_minus_" + str(n), selection_test_df, embedding_test_df

def experiment_dataset_reliability(
    selection_problem,
    selection_source,
    selection_test_source,
    selection_count,
    selection_random_seed,
    selection_reject_minimum,
    selection_overwrite,
    embedding_type,
    embedding_shape,
    embedding_overwrite,
    verbose=True
):
    # get selection set
    selection_df, name = datasets.get_selection_set(
        problem=selection_problem,
        source=selection_source,
        count=selection_count,
        random_seed=selection_random_seed,
        reject_minimum=selection_reject_minimum,
        overwrite=selection_overwrite,
        verbose=verbose
    )

    # create necessary embedding/vector form
    embedding_df = datasets.get_embedding_set(
        selection_df,
        embedding_type=embedding_type,
        output_name=name,
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )

    # TODO: have to return the test stuff too

    return embedding_df, selection_df, name

def experiment_dataset(
    selection_problem,
    selection_test_fold,
    selection_source,
    selection_test_source,
    selection_count,
    selection_random_seed,
    selection_reject_minimum,
    selection_overwrite,
    embedding_type,
    embedding_shape,
    embedding_overwrite,
):
    if selection_problem == "reliability":
        data = experiment_dataset_reliability(
            selection_problem,
            selection_source,
            selection_test_source,
            selection_count,
            selection_random_seed,
            selection_reject_minimum,
            selection_overwrite,
            embedding_type,
            embedding_shape,
            embedding_overwrite
        )
    else:
        data = experiment_dataset_bias(
            selection_problem,
            selection_test_fold,
            selection_count,
            selection_random_seed,
            selection_overwrite,
            embedding_type,
            embedding_shape,
            embedding_overwrite,
        )

    return data


@util.dump_log
def experiment_model(
    selection_problem,
    selection_test_fold,
    selection_source,
    selection_test_source,
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
    model_num,
    verbose=True,
    params=None
):
    # embed_df, sel_df, name = experiment_dataset(
    #     selection_problem,
    #     selection_source,
    #     selection_count,
    #     selection_random_seed,
    #     selection_reject_minimum,
    #     selection_overwrite,
    #     embedding_type,
    #     embedding_shape,
    #     embedding_overwrite,
    #     verbose=verbose
    # )

    # test_selection_df, test_embedding_df = datasets.get_test_embedding_set(selection_problem, selection_source, test_source, selection_count, selection_reject_minimum, selection_random_seed, embedding_type, embedding_shape)

    embed_df, sel_df, name, test_selection_df, test_embedding_df = experiment_dataset(
        selection_problem,
        selection_test_fold,
        selection_source,
        selection_test_source,
        selection_count,
        selection_random_seed,
        selection_reject_minimum,
        selection_overwrite,
        embedding_type,
        embedding_shape,
        embedding_overwrite,
    )

    X = embed_df
    X_test = test_embedding_df
    target_col = ""
    if selection_problem == "reliability":
        target_col = "reliable"
        y = sel_df.reliable
        y_test = test_selection_df.reliable
    elif selection_problem == "biased" or selection_problem == "extreme_biased":
        target_col = "biased"
        y = sel_df.biased
        y_test = test_selection_df.biased

    # pad as needed
    if embedding_shape == "sequence":
        X = lstm.pad_data(X, maxlen=model_maxlen)
        X_test = lstm.pad_data(X_test, maxlen=model_maxlen)

        if model_type == "cnn":
            X = np.reshape(X, (X.shape[0], model_maxlen*300, 1))
            X_test = np.reshape(X_test, (X_test.shape[0], model_maxlen*300, 1))

    #y = keras.utils.to_categorical(y)
    
    name = f'{name}_{model_type}_{model_arch_num}_{model_num}_{model_maxlen}_{model_batch_size}_{model_learning_rate}'
        
    if model_type == "lstm":
        model, history, loss, acc, predictions = lstm.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs, X_test, y_test, name)
    elif model_type == "cnn":
        model, history, loss, acc, predictions = cnn.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs, X_test, y_test, name)
    elif model_type == "nn":
        model, history = nn.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs)
        pass
    elif model_type == "svm":
        pass

    logging.info("%s", str(test_selection_df[target_col].value_counts()))
    print(test_selection_df[target_col].value_counts())

    # turn predictions into dataframe
    #pred = pd.DataFrame({"predicted": predictions})
    #pred.index = test_selection_df.index
    test_selection_df["predicted"] = predictions
    test_selection_df["pred_class"] = round(test_selection_df.predicted).astype(int)
    tp = test_selection_df[(test_selection_df[target_col] == 1) & (test_selection_df.pred_class == 1)].shape[0]
    tn = test_selection_df[(test_selection_df[target_col] == 0) & (test_selection_df.pred_class == 0)].shape[0]
    fp = test_selection_df[(test_selection_df[target_col] == 0) & (test_selection_df.pred_class == 1)].shape[0]
    fn = test_selection_df[(test_selection_df[target_col] == 1) & (test_selection_df.pred_class == 0)].shape[0]

    print(tp, fp)
    print(fn, tn)

    logging.info("tp: %i | fp: %i", tp, fp)
    logging.info("------------------")
    logging.info("fn: %i | tn: %i", fn, tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    logging.info("Precision: %f", precision)
    logging.info("Recall: %f", recall)

    with open("../data/output/" + name + ".json", "w") as outfile:
        results = {"history": history, "testing_loss": loss, "testing_acc": acc, "params": params, "tn": tn, "fn": fn, "tp": tp, "fp": fp, "precision": precision, "recall": recall}
        json.dump(results, outfile)
    with open("../data/output/" + name + "_predictions.pkl", 'wb') as outfile:
        pickle.dump(test_selection_df, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", dest="log_path", default=None)
    parser.add_argument("--temp", dest="temp", default=True)
    parser.add_argument("--experiment", dest="experiment_path", default=None)
    parser.add_argument("--row", dest="experiment_row", type=int, default=None)
    args = parser.parse_args()
    
    util.init_logging(args.log_path)

    if args.experiment_path is not None:
        logging.info("=====================================================")
        if args.experiment_row is not None:
            logging.info("Experiment %s %i started...",args.experiment_path, args.experiment_row)
        else:
            logging.info("Experiment %s started...",args.experiment_path)
        logging.info("=====================================================")

        with open(args.experiment_path, 'r') as infile:
            params = json.load(infile)

        if args.experiment_row is not None:
            params = params[args.experiment_row-1]

        util.TMP_PATH = args.temp

        if params["type"] == "data":
            experiment_dataset(
                params["selection_problem"],
                params["selection_source"],
                params["selection_count"],
                params["selection_random_seed"],
                params["selection_reject_minimum"],
                params["selection_overwrite"],
                params["embedding_type"],
                params["embedding_shape"],
                params["embedding_overwrite"],
                params["verbose"]
            )
        elif params["type"] == "model":
            experiment_model(
                params["selection_problem"],
                params["selection_source"],
                params["selection_count"],
                params["selection_random_seed"],
                params["selection_reject_minimum"],
                params["selection_overwrite"],
                params["embedding_type"],
                params["embedding_shape"],
                params["embedding_overwrite"],
                params["test_source"],
                params["model_type"],
                params["model_arch_num"],
                params["model_layer_sizes"],
                params["model_maxlen"],
                params["model_batch_size"],
                params["model_learning_rate"],
                params["model_epochs"],
                params["model_num"],
                params["verbose"],
                params
                )
    else:
        #experiment_model("reliability", "mbfc", 15000, 13, 500, False, "w2v", "sequence", False, "lstm", 2, (64, 32, 2), 500, 32, .001, 100, 1)
        #experiment_model("reliability", "mbfc", 15000, 13, 500, False, "tfidf", "sequence", False, "lstm", 2, (64, 32, 2), 500, 32, .001, 100, 1, verbose=True)
        datasets.create_selection_set_sources(15000, 500)
        # datasets.get_selection_set(problem="reliability", source="os", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)
        # datasets.get_selection_set(problem="reliability", source="mbfc", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)
        # datasets.get_selection_set(problem="reliability", source="ng", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)
        # datasets.get_selection_set(problem="biased", source="mbfc", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)
        # datasets.get_selection_set(problem="biased", source="as", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)
        # datasets.get_selection_set(problem="extreme_biased", source="mbfc", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)
        # datasets.get_selection_set(problem="extreme_biased", source="as", count=15000, reject_minimum=500, random_seed=13, overwrite=True, verbose=True)


        for i in range(0,10):
            datasets.load_fold(i, 500)
    
