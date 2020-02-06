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
    os.mkdirs(output_path)
except: pass
try:
    os.mkdirs(output_path + "/training_graphs")
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
            figure = vis.make_test_train_plot(results["history"], str(path.name))
            figure.savefig(args.output_path + "/training_graphs/" + str(path.name) + ".png")

df = pd.DataFrame(results_collection)
df.to_csv(output_path + "/compiled.csv")

al_df = pd.DataFrame(results_collection_al)
al_df.to_csv(output_path + "/compiled_al.csv")

al_unseen_df = pd.DataFrame(results_collection_al_unseen)
al_unseen_df.to_csv(output_path + "/compiled_al_unseen.csv")





pathlist_breakdown = Path(args.experiment_results_path + "/breakdown").glob('*.json')

breakdown_results = []
for path in pathlist_breakdown:
    path_in_str = str(path)
    print(f"Processing {path_in_str}...")

    with open(path_in_str, 'r') as infile:
        results = json.load(infile)

    row = {}
    row.update({"source": results["source"], "accuracy": results["accuracy"], "precision": results["precision"], "recall": results["recall"], "tp": results["tp"], "fp": results["fp"], "fn": results["fn"], "tn": results["tn"], "model_num": results["params"]["model_num"], "fold": results["params"]["selection_test_fold"]})
    row["filename"] = path_in_str

    breakdown_results.append(row)
    
df_breakdown = pd.DataFrame(breakdown_results)
df_breakdown.to_csv(args.output_path + "/compiled_breakdown.csv")
