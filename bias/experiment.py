import datasets
import pickle
import logging
import argparse
import util
import json
import keras
import sys
import traceback
import pandas as pd
import numpy as np

import lstm
import cnn
import nn

# TODO: pass tuple of counts instead of tp fp etc
#def confusion_analysis(tp, fp, tn, fn, name, history, loss, acc, params, source=False):
def confusion_analysis(counts, name, history, loss, acc, params, source=False):
    if source:
        logging.info("Confusion analysis for %s", source)

    binary = True
    if len(counts) > 4:
        binary = False

    results = {}
    if source:
        results = {"source": source, "history": history, "testing_loss": loss, "testing_acc": acc, "params": params}
    else:
        results = {"history": history, "testing_loss": loss, "testing_acc": acc, "params": params}

    if binary:
        tp, fp, fn, tn = counts
        
        logging.info("tp: %i | fp: %i", tp, fp)
        logging.info("------------------")
        logging.info("fn: %i | tn: %i", fn, tn)

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else: 
            precision = 0

        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        logging.info("Precision: %f", precision)
        logging.info("Recall: %f", recall)
        logging.info("Accuracy: %f", accuracy)
        
        results.update({"tn":tp, "fp":fp, "fn":fn, "tn":tn, "precision": precision, "recall": recall, "accuracy": accuracy})
    else:
        ll, lc, lr, cl, cc, cr, rl, rc, rr = counts

        logging.info("ll: %i | lc: %i | lr: %i", ll, lc, lr)
        logging.info("------------------------------")
        logging.info("cl: %i | cc: %i | cr: %i", cl, cc, cr)
        logging.info("------------------------------")
        logging.info("rl: %i | rc: %i | rr: %i", rl, rc, rr)

        accuracy = (ll + cc + rr) / (ll + lc + lr + cl + cc + cr + rl + rc + rr)

        logging.info("Accuracy: %f", accuracy)
        
        results.update({"ll":ll, "lc":lc, "lr":lr, "cl":cl, "cc":cc, "cr":cr, "rl":rl, "rc":rc, "rr":rr, "accuracy": accuracy})
        

    filename = name + ".json"
    if source:
        filename = name + "_" + source + ".json"

    with open("../data/output/" + filename, "w") as outfile:
        json.dump(results, outfile)
    
# predicted, actual
def calculate_counts(df, target_col, pred_val, target_val):
    count = df[(df[target_col] == target_val) & (df.pred_class == pred_val)].shape[0]
    return count

# NOTE: row is predicted, col is actual   (in second case, predicted|actual)
# tp, fp, fn, tn (pp, pn, fp, ff)
# ll, lc, lr, cl, cc, cr, rl, rc, rr
def calculate_cm_counts(df, target_col, binary=True):
    counts = [] 
    if binary:
        tp = calculate_counts(df, target_col, 1, 1)
        fp = calculate_counts(df, target_col, 1, 0)
        fn = calculate_counts(df, target_col, 0, 1)
        tn = calculate_counts(df, target_col, 0, 0)
        
        counts = [tp, fp, fn, tn]
    else:
        ll = calculate_counts(df, target_col, -1, -1)
        lc = calculate_counts(df, target_col, -1, 0)
        lr = calculate_counts(df, target_col, -1, 1)
        
        cl = calculate_counts(df, target_col, 0, -1)
        cc = calculate_counts(df, target_col, 0, 0)
        cr = calculate_counts(df, target_col, 0, 1)
        
        rl = calculate_counts(df, target_col, 1, -1)
        rc = calculate_counts(df, target_col, 1, 0)
        rr = calculate_counts(df, target_col, 1, 1)
        
        counts = [ll, lc, lr, cl, cc, cr, rl, rc, rr]

    return counts
        

@util.dump_log
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

    name = "{0}_{1}_{2}".format(selection_problem, selection_random_seed, selection_count) 

    binary = True
    if selection_problem == "bias_direction":
        binary = False

    selection_df, selection_test_df = datasets.load_folds(selection_test_fold, selection_count, binary, selection_overwrite)

    embedding_df = datasets.get_embedding_set(
        selection_df,
        embedding_type=embedding_type,
        output_name=name + "_minusfold" + str(selection_test_fold),
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )
    
    embedding_test_df = datasets.get_embedding_set(
        selection_test_df,
        embedding_type=embedding_type,
        output_name=name + "_fold" + str(selection_test_fold),
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )

    return embedding_df, selection_df, name + "fold_minus_" + str(selection_test_fold), selection_test_df, embedding_test_df

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

@util.dump_log
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
    verbose=True
):
    print(selection_problem)
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
        embed_df, sel_df, name, test_selection_df, test_embedding_df = experiment_dataset_bias(
            selection_problem,
            selection_test_fold,
            selection_count,
            selection_random_seed,
            selection_overwrite,
            embedding_type,
            embedding_shape,
            embedding_overwrite,
        )

    return embed_df, sel_df, name, test_selection_df, test_embedding_df

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
    elif selection_problem == "biased" or selection_problem == "extreme_biased" or selection_problem == "bias_direction": # NOTE: unsure if this is where bias_direction should go?
        target_col = "biased"
        y = sel_df.biased
        y_test = test_selection_df.biased

    # pad as needed
    data_width=0
    # TODO: how to determine data width for avg?
    if embedding_shape == "sequence":
        X = lstm.pad_data(X, maxlen=model_maxlen)
        X_test = lstm.pad_data(X_test, maxlen=model_maxlen)

        # TODO: 300 actually needs to be width (num cols) of dataset
        data_width = X.shape[-1]
        if model_type == "cnn":
            X = np.reshape(X, (X.shape[0], model_maxlen*data_width, 1))
            X_test = np.reshape(X_test, (X_test.shape[0], model_maxlen*data_width, 1))

    #y = keras.utils.to_categorical(y)
    
    name = f'{name}_{model_type}_{model_arch_num}_{model_num}_{model_maxlen}_{model_batch_size}_{model_learning_rate}'
        
    if model_type == "lstm":
        model, history, loss, acc, predictions = lstm.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs, X_test, y_test, name, data_width)
    elif model_type == "cnn":
        model, history, loss, acc, predictions = cnn.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs, X_test, y_test, name)
    elif model_type == "nn":
        model, history = nn.train_test(X, y, model_arch_num, model_layer_sizes, model_maxlen, model_batch_size, model_learning_rate, model_epochs)
        pass
    elif model_type == "svm":
        pass
    print("Training done")

    logging.info("%s", str(test_selection_df[target_col].value_counts()))
    print(test_selection_df[target_col].value_counts())

    # turn predictions into dataframe
    #pred = pd.DataFrame({"predicted": predictions})
    #pred.index = test_selection_df.index
    test_selection_df["predicted"] = predictions
    test_selection_df["pred_class"] = round(test_selection_df.predicted).astype(int)

    overall_counts = [] 
    if selection_problem != "bias_direction":
        overall_counts = calculate_cm_counts(test_selection_df, target_col, binary=True)
    else:
        overall_counts = calculate_cm_counts(test_selection_df, target_col, binary=False)
        
    
    # tp = test_selection_df[(test_selection_df[target_col] == 1) & (test_selection_df.pred_class == 1)].shape[0]
    # tn = test_selection_df[(test_selection_df[target_col] == 0) & (test_selection_df.pred_class == 0)].shape[0]
    # fp = test_selection_df[(test_selection_df[target_col] == 0) & (test_selection_df.pred_class == 1)].shape[0]
    # fn = test_selection_df[(test_selection_df[target_col] == 1) & (test_selection_df.pred_class == 0)].shape[0]

    logging.info("Overall confusion analysis")
    confusion_analysis(overall_counts, name, history, loss, acc, params, False)
    logging.info("Overall analysis complete")



    groups = test_selection_df.groupby(test_selection_df.source)
    logging.info("There are %i groups", len(groups))


    for group_name, group in groups:
        logging.info("Next group %s", name)

        group_counts = []
        if selection_problem != "bias_direction":
            group_counts = calculate_cm_counts(group, target_col, binary=True)
        else:
            group_counts = calculate_cm_counts(group, target_col, binary=False)
            

        confusion_analysis(group_counts, name + "_persource", history, loss, acc, params, source=group_name)

    with open("../data/output/" + name + "_predictions.pkl", 'wb') as outfile:
        pickle.dump(test_selection_df, outfile)


#if __name__ == "__main__":
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
            params["selection_test_fold"],
            params["selection_source"],
            params["selection_test_source"],
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
            params["selection_test_fold"],
            params["selection_source"],
            params["selection_test_source"],
            params["selection_count"],
            params["selection_random_seed"],
            params["selection_reject_minimum"],
            params["selection_overwrite"],
            params["embedding_type"],
            params["embedding_shape"],
            params["embedding_overwrite"],
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
        datasets.load_fold(i, 1000, True, True)

