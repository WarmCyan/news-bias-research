#!/bin/python3
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import sys
sys.path.append('..')

from bias import vis

parser = argparse.ArgumentParser()
parser.add_argument("-e", dest="experiment_results_path", default=None, help="experiment results folder", required=True)
parser.add_argument("-o", dest="output_path", help="compiled output path", required=True)
args = parser.parse_args()


results_collection = []
results_collection_al = []
results_collection_al_unseen = []

output_path = "../data/output/compiled/" + args.output_path

try:
    os.makedirs(output_path)
except: pass
try:
    os.makedirs(output_path + "/training_graphs")
except: pass

results_paths = args.experiment_results_path.split(",")

for result_path in results_paths:
    actual_path = "../data/output/" + result_path
    # pathlist = Path(args.experiment_results_path).glob('**/*.json')
    pathlist = Path(actual_path).glob('*.json')
    for path in pathlist:
        path_in_str = str(path)
        print(f"Processing {path_in_str}...")

        al = False
        al_unseen = False

        if path_in_str[-7:] == "al.json":
            print("Recognized AL data")
            al = True
        elif path_in_str[-14:] == "al_unseen.json":
            print("Recognized AL unseen source data")
            al_unseen = True
        elif path_in_str[-24:] == "al_unseensourcelist.json":
            print("Recognized an unseen source list")
            continue

        with open(path_in_str, 'r') as infile:
            results = json.load(infile)

        row = results["params"]
        if "testing_loss" in results:
            row["testing_loss"] = results["testing_loss"]
        if "testing_acc" in results:
            row["testing_acc"] = results["testing_acc"]
        # TODO: tn etc only for two way classification, will need to fix for threeway
        row["tn"] = results["tn"]
        row["tp"] = results["tp"]
        row["fn"] = results["fn"]
        row["fp"] = results["fp"]
        row["precision"] = results["precision"]
        row["recall"] = results["recall"]
        row["accuracy"] = results["accuracy"]
        row["experiment_tag"] = results["experiment_tag"]
        #row["filename"] = path_in_str

        if al:
            results_collection_al.append(row)
        elif al_unseen:
            results_collection_al_unseen.append(row)
        else:
            results_collection.append(row)
            #figure = vis.make_test_train_plot(results["history"], str(path.name))
            #figure.savefig(args.output_path + "/training_graphs/" + str(path.name) + ".png")

df = pd.DataFrame(results_collection)
df.to_csv(output_path + "/compiled.csv")

al_df = pd.DataFrame(results_collection_al)
al_df.to_csv(output_path + "/compiled_al.csv")

al_unseen_df = pd.DataFrame(results_collection_al_unseen)
al_unseen_df.to_csv(output_path + "/compiled_al_unseen.csv")


# TODO: aggregate stats next 
experiment_names = list(set(df.experiment_tag))

aggregate = []

# group the compiled by problem (TODO) and then fold
groups = df.groupby(df["selection_test_fold"])
for group_name, group in groups:
    print(group_name)
    row = {}
    row["fold"] = group_name
    
    # get each experiment tag
    for name in experiment_names:
        if group[group.experiment_tag == name].shape[0] > 0:
            row[name] = group[group.experiment_tag == name].iloc[0].accuracy

    aggregate.append(row)

aggregate_df = pd.DataFrame(aggregate)
#aggregate_df.columns = ["fold", *experiment_names]
aggregate_df.to_csv(output_path + "/aggregate.csv")



aggregate_al = []

groups = al_df.groupby(al_df["selection_test_fold"])
for group_name, group in groups:
    print(group_name)
    row = {}
    row["fold"] = group_name
    
    # get each experiment tag
    for name in experiment_names:
        if group[group.experiment_tag == name].shape[0] > 0:
            row[name] = group[group.experiment_tag == name].iloc[0].accuracy

    aggregate_al.append(row)

aggregate_al_df = pd.DataFrame(aggregate_al)
aggregate_al_df.to_csv(output_path + "/aggregate_al.csv")


aggregate_al_unseen = []

groups = al_unseen_df.groupby(al_unseen_df["selection_test_fold"])
for group_name, group in groups:
    print(group_name)
    row = {}
    row["fold"] = group_name
    
    # get each experiment tag
    for name in experiment_names:
        if group[group.experiment_tag == name].shape[0] > 0:
            row[name] = group[group.experiment_tag == name].iloc[0].accuracy

    aggregate_al_unseen.append(row)

aggregate_al_unseen_df = pd.DataFrame(aggregate_al_unseen)
aggregate_al_unseen_df.to_csv(output_path + "/aggregate_al_unseen.csv")


# ===========================================
# Persource analysis
# ===========================================


bias_detail_df = pd.read_csv("../data/cache/bias_folds_detail.csv")
sentics_detail_df = pd.read_csv("../data/cache/biased_aggregate_sentics.csv")


breakdown_results = []
for result_path in results_paths:
    actual_path = "../data/output/" + result_path
        
    pathlist_breakdown = Path(actual_path + "/persource").glob('*.json')

    for path in pathlist_breakdown:
        path_in_str = str(path)
        print(f"Processing {path_in_str}...")

        with open(path_in_str, 'r') as infile:
            results = json.load(infile)

        row = {}
        row["experiment_tag"] = results["experiment_tag"]
        row.update({"source": results["source"], "accuracy": results["accuracy"], "precision": results["precision"], "recall": results["recall"], "tp": results["tp"], "fp": results["fp"], "fn": results["fn"], "tn": results["tn"], "model_num": results["params"]["model_num"], "fold": results["params"]["selection_test_fold"]})
        row["filename"] = path_in_str

        breakdown_results.append(row)
    
df_breakdown = pd.DataFrame(breakdown_results)
df_breakdown.to_csv(output_path + "/compiled_breakdown.csv")


aggregate_breakdown = []
groups = df_breakdown.groupby(df_breakdown["source"])
for group_name, group in groups:
    print(group_name)
    row = {}
    row["source"] = group_name
    
    # get each experiment tag
    for name in experiment_names:
        if group[group.experiment_tag == name].shape[0] > 0:
            row[name] = group[group.experiment_tag == name].iloc[0].accuracy

    try:
        row.update(dict(bias_detail_df[bias_detail_df.Source == group_name].iloc[0]))
        row.update(dict(sentics_detail_df[sentics_detail_df.source == group_name].iloc[0]))
    except:
        print("WARNING, could not add bias and sentics detail")

    aggregate_breakdown.append(row)

aggregate_breakdown_df = pd.DataFrame(aggregate_breakdown)
#aggregate_df.columns = ["fold", *experiment_names]
aggregate_breakdown_df.to_csv(output_path + "/aggregate_breakdown.csv")


# ===========================================
# Persource analysis (article level)
# ===========================================



al_breakdown_results = []
for result_path in results_paths:
    actual_path = "../data/output/" + result_path
        
    pathlist_breakdown = Path(actual_path + "/alpersource").glob('*.json')

    for path in pathlist_breakdown:
        path_in_str = str(path)
        print(f"Processing {path_in_str}...")

        with open(path_in_str, 'r') as infile:
            results = json.load(infile)

        row = {}
        row["experiment_tag"] = results["experiment_tag"]
        row.update({"source": results["source"], "accuracy": results["accuracy"], "precision": results["precision"], "recall": results["recall"], "tp": results["tp"], "fp": results["fp"], "fn": results["fn"], "tn": results["tn"], "model_num": results["params"]["model_num"], "fold": results["params"]["selection_test_fold"]})
        row["filename"] = path_in_str

        al_breakdown_results.append(row)
    
df_breakdown_al = pd.DataFrame(al_breakdown_results)
df_breakdown_al.to_csv(output_path + "/compiled_breakdown.csv")


aggregate_breakdown = []
groups = df_breakdown_al.groupby(df_breakdown_al["source"])
for group_name, group in groups:
    print(group_name)
    row = {}
    row["source"] = group_name
    
    # get each experiment tag
    for name in experiment_names:
        if group[group.experiment_tag == name].shape[0] > 0:
            row[name] = group[group.experiment_tag == name].iloc[0].accuracy

    aggregate_breakdown.append(row)

aggregate_breakdown_df = pd.DataFrame(aggregate_breakdown)
#aggregate_df.columns = ["fold", *experiment_names]
aggregate_breakdown_df.to_csv(output_path + "/aggregate_breakdown_al.csv")
