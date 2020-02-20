#!/bin/python3
import sys
import json
import pandas as pd
from tqdm import tqdm

import argparse
from pathlib import Path

sys.path.append('..')
from bias import util

parser = argparse.ArgumentParser()
parser.add_argument("-e", dest="experiment_results_path", default=None, help="experiment results folder", required=True)
#parser.add_argument("-o", dest="output_path", help="compiled output path", required=True)
args = parser.parse_args()

# predicted, actual
def calculate_counts(df, target_col, pred_val, target_val):
    count = df[(df[target_col] == target_val) & (df.pred_class == pred_val)].shape[0]
    return count

def calculate_cm_counts(df, target_col, binary=True):
    counts = [] 
    if binary:
        tp = calculate_counts(df, target_col, 1, 1)
        fp = calculate_counts(df, target_col, 1, 0)
        fn = calculate_counts(df, target_col, 0, 1)
        tn = calculate_counts(df, target_col, 0, 0)
        
        counts = [tp, fp, fn, tn]
    else:
        ll = calculate_counts(df, target_col, 0, 0)
        lc = calculate_counts(df, target_col, 0, 1)
        lr = calculate_counts(df, target_col, 0, 2)
        
        cl = calculate_counts(df, target_col, 1, 0)
        cc = calculate_counts(df, target_col, 1, 1)
        cr = calculate_counts(df, target_col, 1, 2)
        
        rl = calculate_counts(df, target_col, 2, 0)
        rc = calculate_counts(df, target_col, 2, 1)
        rr = calculate_counts(df, target_col, 2, 2)
        
        counts = [ll, lc, lr, cl, cc, cr, rl, rc, rr]

    return counts

# how did it do [average] in al (when it was seen in 9 folds) versus how it did when it was in unseen


actual_path = "../data/output/" + args.experiment_results_path


# get all the sources
mbc_df = util.load_scraped_mpc()
al_sources = list(set(mbc_df.Source))

fold_unseen = {}


# need to get selection tag and then load the folds
#util.load_fold_divisions_dataset(selection)

folds = None
sources_list = []
fold_seen = {}




source_scores = {} # key for source name, underneath "unseen", and "seen" (seen should be array of 9 scores)


# grab all the unseensourcelists

unseenlist_pathlist = Path(actual_path).glob('*unseensourcelist.json')
for path in tqdm(unseenlist_pathlist):
    path = str(path)
    related_path = path[:-21] + "unseen.json"
    with open(path, 'r') as infile:
        unseenlist = json.load(infile)

    with open(related_path, 'r') as infile:
        unseen_data = json.load(infile)

    fold_num = str(unseen_data["params"]["selection_test_fold"])

    # get fold data if we don't have it
    if folds is None:
        sel_tag = unseen_data["params"]["selection_tag"]
        folds = util.load_fold_divisions_dataset(sel_tag)
        for dictionary in folds:
            sources_list.extend(dictionary["left"])
            sources_list.extend(dictionary["right"])
            sources_list.extend(dictionary["center"])

        for dictionary in folds:
            fold_seen[fold_num] = []
            for source in sources_list:
                if source not in dictionary["left"] and source not in dictionary["right"] and source not in dictionary["center"]:
                    fold_seen[fold_num].append(source)
                    
        
    fold_unseen[fold_num] = unseenlist

    df_path = path[:-24] + "predictionsal.pkl"
    predictions_df = pd.read_pickle(df_path)
    sources = list(set(predictions_df.Source))


    #for source in unseenlist:
    for source in sources:
        if source in sources_list:   
            # meaning this is one we have data both for and not, depending on fold

            # get the score
            tp, fp, fn, tn = calculate_cm_counts(predictions_df[predictions_df.Source == source], "biased")
            acc = (tp + tn) / (tp + fp + fn + tn)
            
            if source not in source_scores:
                source_scores[source] = {"seen":[], "unseen":0}

            # it's unseen here
            if source in unseenlist:
                source_scores[source]["unseen"] = acc
                source_scores[source]["count"] = (tp + fp + fn + tn)
            else:
                source_scores[source]["seen"].append(acc)

#print(source_scores)

print("Source".ljust(25), "Seen".ljust(25), "Unseen".ljust(25))
for source in source_scores:
    seen_avg = 0
    for val in source_scores[source]["seen"]:
        seen_avg += val
    seen_avg /= 9

    print(
        source.ljust(25),
        str(seen_avg).ljust(25),
        str(source_scores[source]["unseen"]).ljust(25),
        str(source_scores[source]["count"]).ljust(25)
    )
    
