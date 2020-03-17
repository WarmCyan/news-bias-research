#!/bin/python3
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import sys
import re
sys.path.append('..')

from bias import vis

parser = argparse.ArgumentParser()
parser.add_argument("-e", dest="experiment_results_path", default=None, help="experiment results folder", required=True)
parser.add_argument("-o", dest="output_path", help="compiled output path", required=True)
parser.add_argument("--caption", dest="caption", default="CAPTION")
parser.add_argument("--column-replacements", dest="columns")
parser.add_argument("--final", dest="final", action="store_true") #
parser.add_argument("--ALL", dest="ALL", action="store_true")
parser.add_argument("--row-order", dest="row_order")
args = parser.parse_args()


THREEWAY = False
BIAS = True
ALL = args.ALL

results_collection = []
results_collection_al = []
results_collection_al_unseen = []

output_path = "../data/output/compiled/" + args.output_path
final_table_output_path = "../data/output/tables/"

try:
    os.makedirs(output_path)
except: pass
try:
    os.makedirs(output_path + "/training_graphs")
except: pass
try:
    os.makedirs(final_table_output_path)
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
        if row["selection_problem"] == "bias_direction":
            THREEWAY = True
            row["ll"] = results["ll"]
            row["lc"] = results["lc"]
            row["lr"] = results["lr"]
            row["cl"] = results["cl"]
            row["cc"] = results["cc"]
            row["cr"] = results["cr"]
            row["rl"] = results["rl"]
            row["rc"] = results["rc"]
            row["rr"] = results["rr"]

            row["l_rec"] = results["ll"]/(results["ll"] + results["cl"] + results["rl"])
            row["c_rec"] = results["cc"]/(results["lc"] + results["cc"] + results["rc"])
            row["r_rec"] = results["rr"]/(results["lr"] + results["cr"] + results["rr"])

            row["l_prec"] = results["ll"]/(results["ll"] + results["lc"] + results["lr"])
            row["c_prec"] = results["cc"]/(results["cl"] + results["cc"] + results["cr"])
            row["r_prec"] = results["rr"]/(results["rl"] + results["rc"] + results["rr"])

            row["l_f1"] = 2*(row["l_prec"]*row["l_rec"])/(row["l_prec"]+row["l_rec"])
            row["c_f1"] = 2*(row["c_prec"]*row["c_rec"])/(row["c_prec"]+row["c_rec"])
            row["r_f1"] = 2*(row["r_prec"]*row["r_rec"])/(row["r_prec"]+row["r_rec"])

            row["total"] = row["ll"] + row["lc"] + row["lr"] + row["cl"] + row["cc"] + row["cr"] + row["rl"] + row["rc"] + row["rr"]
        else:
            if row["selection_problem"] == "reliability":
                BIAS = False
            row["tn"] = results["tn"]
            row["tp"] = results["tp"]
            row["fn"] = results["fn"]
            row["fp"] = results["fp"]
            row["precision"] = results["precision"]
            row["recall"] = results["recall"]
            row["f1"] = 2*(results["precision"]*results["recall"])/(results["precision"]+results["recall"])
        row["accuracy"] = results["accuracy"]
        row["experiment_tag"] = results["experiment_tag"]
        row["path"] = path_in_str
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
        if THREEWAY:
            row.update({"source": results["source"], "accuracy": results["accuracy"], "ll": results["ll"], "lc": results["lc"], "lr": results["lr"], "cl": results["cl"], "cc": results["cc"], "cr": results["cr"], "rl": results["rl"], "rc": results["rc"], "rr": results["rr"], "model_num": results["params"]["model_num"], "fold": results["params"]["selection_test_fold"]})
        else:
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
        if THREEWAY:
            row.update({"source": results["source"], "accuracy": results["accuracy"], "ll": results["ll"], "lc": results["lc"], "lr": results["lr"], "cl": results["cl"], "cc": results["cc"], "cr": results["cr"], "rl": results["rl"], "rc": results["rc"], "rr": results["rr"], "model_num": results["params"]["model_num"], "fold": results["params"]["selection_test_fold"]})
        else:
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


# ===========================================
# Multi-run analyses and better tables
# ===========================================


# tp, fp, fn, tn (pp, pn, fp, ff)
# ll, lc, lr, cl, cc, cr, rl, rc, rr
def make_latex_cm_table(data, output='table', caption="Caption", label="tab:my_label", three_way=False, name_true="Biased", name_false="Unbiased"):
    runstring = " ".join(sys.argv[:])
    beginning = '''
\\begin{table}[h!] 
    \\centering'''

    col_count = 2

    if three_way:
        col_count = 3

        
    beginning += '''
    \\begin{tabular}{cr|''' + "c"*col_count + '''}
'''

    tabletext = '''\\toprule\n
    & & \\multicolumn{''' + str(col_count) + '''}{c}{Actual} \\\\ 
    '''

    if three_way:
        tabletext += '''
        & & Left & Center & Right \\\\
    \\midrule
        \\multirow{3}{*}{Predicted}
'''
#"{0:.2f}".format(data)
        tabletext += "& Left & \\cellcolor{lightgray}" + "{0:.0f}".format(data["ll"]) + " & " + "{0:.0f}".format(data["lc"]) + " & " + "{0:.0f}".format(data["lr"]) + " \\\\ \n"
        tabletext += "& Center & " + "{0:.0f}".format(data["cl"]) + " & \\cellcolor{lightgray}" + "{0:.0f}".format(data["cc"]) + " & " + "{0:.0f}".format(data["cr"]) + " \\\\ \n"
        tabletext += "& Right & " + "{0:.0f}".format(data["rl"]) + " & " + "{0:.0f}".format(data["rc"]) + " & \\cellcolor{lightgray}" + "{0:.0f}".format(data["rr"]) + " \\\\ \n"

    else:
        tabletext += '''
        & & ''' + name_true + ''' & ''' + name_false + ''' \\\\
    \\midrule
        \\multirow{2}{*}{Predicted}
'''
        tabletext += "& " + name_true + " & \\cellcolor{lightgray}" + "{0:.2f}".format(data["tp"]) + " & " + "{0:.2f}".format(data["fp"]) + " \\\\ \n"
        tabletext += "& " + name_false + " & " + "{0:.2f}".format(data["fn"]) + " & \\cellcolor{lightgray}" + "{0:.2f}".format(data["tn"]) + " \\\\ \n"

    ending = '''
    \\bottomrule
    \\end{tabular}
    \\caption{''' + caption + '''}
    \\label{''' + label + '''}
\\end{table}'''

    latex_text = "%" + runstring + beginning + tabletext + ending
    
    print(latex_text)
    
    with open(output_path + "/" + output + ".tex", 'w') as outfile:
        outfile.write(latex_text)

    if args.final:
        with open(final_table_output_path + output + ".tex", 'w') as outfile:
            outfile.write(latex_text)
    

def make_latex_table(df, output='table', caption="Caption", label="tab:my_label", alignment=None, bold=True):
    global output_path
    global args
    global ALL
    global THREEWAY

    if ALL:
        bold = False
    
    runstring = " ".join(sys.argv[:])
    beginning = '''
\\begin{table}[h!] 
    \\centering'''

    ending = '''
    \\bottomrule
    \\end{tabular}
    \\caption{''' + caption + '''}
    \\label{''' + label + '''}
\\end{table}'''

    if alignment is None:
        col_count = df.shape[1]
        alignment = "r|" + "c"*(col_count)
        #alignment = alignment[:-1]
        #alignment = "|" + alignment

    beginning += '''
    \\begin{tabular}{''' + alignment + '''}
'''

    tabletext = "\\toprule\n"

    for col in df.columns:
        tabletext += " & " + col.replace("_",'\_')
    tabletext += "\\\\"
    tabletext += "\n\\midrule"

    for index, row in df.iterrows():
        if str(index) in ["ll", "lr", "lc", "cl", "cc", "cr", "rl", "rc", "rr", "tp", "tn", "fp", "fn"]:
            continue

        row_label = index

        rowmax=row.iloc[0]
        if bold:
            for data in row:
                if data > rowmax:
                    rowmax = data
        
        tabletext += "\n" + index + " "
        for index,data in enumerate(row):

            if ALL:
                if index >= 4 and not THREEWAY:
                    tabletext += "& " + str(int(data)) + " "
                else:
                    tabletext += "& " + "{0:.2f}".format(data) + " "
            else:
                if bold and data == rowmax:
                    tabletext += "& \\textbf{" + "{0:.2f}".format(data) + "} "
                else: 
                    tabletext += "& " + "{0:.2f}".format(data) + " "
            
        if THREEWAY and (row_label == "Accuracy" or row_label == "R Precision" or row_label == "R Recall"):
            tabletext += "\\\\\n\\midrule\n"
        else:
            tabletext += "\\\\\n"
    

    latex_text = "%" + runstring + beginning + tabletext + ending
    
    print(latex_text)
    
    with open(output_path + "/" + output + ".tex", 'w') as outfile:
        outfile.write(latex_text)

    if args.final:
        with open(final_table_output_path + output + ".tex", 'w') as outfile:
            outfile.write(latex_text)
    


def make_combined_table(df):
    global THREEWAY
    global ALL
    table_rows = []
    
    groups = df.groupby(df.experiment_tag)
    for group_name, group in groups:
        print("="*30, group_name)
        row = {
            "Name": group_name,
            "Accuracy": group.accuracy.mean()*100,
        }
        
        if THREEWAY:
            # for index, localrow in group.iterrows():
            #     total = localrow["ll"] + localrow["lc"] + localrow["lr"] + localrow["cl"] + localrow["cc"] + localrow["cr"] + localrow["rl"] + localrow["rc"] + localrow["rr"]
            #     group.loc[index, "ll"] = localrow["ll"] / total
            #     group.loc[index, "lc"] = localrow["lc"] / total
            #     group.loc[index, "lr"] = localrow["lr"] / total
            #     group.loc[index, "cl"] = localrow["cl"] / total
            #     group.loc[index, "cc"] = localrow["cc"] / total
            #     group.loc[index, "cr"] = localrow["cr"] / total
            #     group.loc[index, "rl"] = localrow["rl"] / total
            #     group.loc[index, "rc"] = localrow["rc"] / total
            #     group.loc[index, "rr"] = localrow["rr"] / total
               
            row.update({
                "ll": group.ll.mean(),
                "lc": group.lc.mean(),
                "lr": group.lr.mean(),
                "cl": group.cl.mean(),
                "cc": group.cc.mean(),
                "cr": group.cr.mean(),
                "rl": group.rl.mean(),
                "rc": group.rc.mean(),
                "rr": group.rr.mean(),
            })

            row.update({
                "L Precision": group.l_prec.mean()*100,
                "C Precision": group.c_prec.mean()*100,
                "R Precision": group.r_prec.mean()*100,
                "L Recall": group.l_rec.mean()*100,
                "C Recall": group.c_rec.mean()*100,
                "R Recall": group.r_rec.mean()*100,
                "L F1": group.l_f1.mean()*100,
                "C F1": group.c_f1.mean()*100,
                "R F1": group.r_f1.mean()*100,
            })
        else:
            # for index, localrow in group.iterrows():
            #     total = localrow["tp"] + localrow["fp"] + localrow["fn"] + localrow["tn"]
            #     group.loc[index, "tp"] = localrow["tp"] / total
            #     group.loc[index, "fp"] = localrow["fp"] / total
            #     group.loc[index, "fn"] = localrow["fn"] / total
            #     group.loc[index, "tn"] = localrow["tn"] / total

            row.update({
                "Precision": group.precision.mean()*100,
                "Recall": group.recall.mean()*100,
                "F1": group.f1.mean()*100,
                "tp": group.tp.mean(),
                "fp": group.fp.mean(),
                "fn": group.fn.mean(),
                "tn": group.tn.mean(),
            })
            
        print(row)
        table_rows.append(row)
    combined_df = pd.DataFrame(table_rows)
    print(combined_df)
    combined_df.index = combined_df.Name
    combined_df = combined_df.drop(columns=["Name"])
    
    if ALL:
        combined_df = combined_df.rename(columns={"tp":"TP","fp":"FP","fn":"FN","tn":"TN", "ll":"LL","lc":"LC","lr":"LR","cl":"CL","cc":"CC","cr":"CR","rl":"RL", "rc":"RC", "rr":"RR"})
        combined_df = combined_df.drop(columns=["LL", "LC", "LR", "CL", "CC", "CR", "RL", "RC", "RR"])
        combined_df = combined_df.reindex(args.row_order.split(","))
        return combined_df
    else:
        return combined_df.transpose()


def fix_column_names(df, replacement_string):
    global ALL
    replacements = replacement_string.split(",")

    col_redefs = {}
    order = []

    for replacement in replacements:
        parts = replacement.split("=")
        tag, newname = parts

        col_redefs[tag] = newname
        order.append(newname)

    if ALL:
        df = df.rename(index=col_redefs)
        print(col_redefs)
        print(order)
        print("#"*50)
        print(df)
        df = df.reindex(order)
        print("8"*50)
        print(df)
    else:
        df = df.rename(columns=col_redefs)
        df = df[order]
    return df
    

combined_table = []
combined_table_al = []
combined_table_unseen = []

name_prefix = args.output_path

combined_df = make_combined_table(df)
combined_df_al = make_combined_table(al_df)
print("*"*50)
print(combined_df_al)
combined_df_al_unseen = make_combined_table(al_unseen_df)

# print(combined_df_al)

if args.columns is not None:
    combined_df = fix_column_names(combined_df, args.columns)
    combined_df_al = fix_column_names(combined_df_al, args.columns)
    combined_df_al_unseen = fix_column_names(combined_df_al_unseen, args.columns)

sl_name = name_prefix + "_sl"
make_latex_table(combined_df, sl_name, caption=args.caption + " (Source-level.)", label="tab:" + sl_name)

al_name = name_prefix + "_al"
make_latex_table(combined_df_al, al_name, caption=args.caption + " (Article-level.)", label="tab:" + al_name)

al_unseen_name = name_prefix + "_al_unseen"
make_latex_table(combined_df_al_unseen, al_unseen_name, caption=args.caption + " (Article-level, from only unseen sources.)", label="tab:" + al_unseen_name)


print(combined_df_al)

if ALL:
    exit()

for index, row in combined_df_al.transpose().iterrows():
    safe_index = re.sub(r'[^\w]', '', str(index))

    labelT = "Biased"
    labelF = "Unbiased"

    if not BIAS:
        labelT = "Reliable"
        labelF = "Unreliable"
        
    
    make_latex_cm_table(row, name_prefix + "_" + str(safe_index), caption=args.caption + " (" + str(index) + ")", three_way=THREEWAY, name_true=labelT, name_false=labelF)
