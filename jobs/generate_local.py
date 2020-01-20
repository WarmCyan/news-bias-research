import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="name", default=None, help="name")
parser.add_argument("-e", dest="experiment_path", default=None, help="experiment file name", required=True)
parser.add_argument("-r", dest="experiment_row", default=None, help="row")
parser.add_argument("-m", dest="memory", default="96000", help="memory (96000)")
parser.add_argument("-t", dest="time", default="6:00:00", help="time (6:00:00)")
args = parser.parse_args()

name = args.name

if name is None:
    name = args.experiment_path
    if args.experiment_row is not None:
        name += "_" + args.experiment_row

common_lines = [
    "#!/bin/bash",
    "TMP=$(mktemp -d)",
    "pushd ../bias",
]

experiment_line = "python experiment.py --experiment ../experiments/" + args.experiment_path + ".json --temp $TMP"
if args.experiment_row is not None:
    experiment_line += " --row " + args.experiment_row
experiment_line += " --log ../logs/" + name + ".log"

common_lines.append(experiment_line)
common_lines.append("popd")

for i in range(0, len(common_lines)):
    common_lines[i] += "\n"

with open(name + '_local.sh', 'w') as outfile:
    outfile.writelines(common_lines)

print(name + ".sh")
