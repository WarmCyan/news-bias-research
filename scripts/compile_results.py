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

try:
    os.mkdir(args.output_path)
except: pass
try:
    os.mkdir(args.output_path + "/training_graphs")
except: pass

# pathlist = Path(args.experiment_results_path).glob('**/*.json')
pathlist = Path(args.experiment_results_path).glob('*.json')
for path in pathlist:
    path_in_str = str(path)
    print(f"Processing {path_in_str}...")

    with open(path_in_str, 'r') as infile:
        results = json.load(infile)

    row = results["params"]
    row["testing_loss"] = results["testing_loss"]
    row["testing_acc"] = results["testing_acc"]
    row["tn"] = results["tn"]
    row["tp"] = results["tp"]
    row["fn"] = results["fn"]
    row["fp"] = results["fp"]
    row["precision"] = results["precision"]
    row["recall"] = results["recall"]
    row["filename"] = path_in_str

    results_collection.append(row)

    figure = vis.make_test_train_plot(results["history"], str(path.name))
    figure.savefig(args.output_path + "/training_graphs/" + str(path.name) + ".png")

df = pd.DataFrame(results_collection)
df.to_csv(args.output_path + "/compiled.csv")

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
