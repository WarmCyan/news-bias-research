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
    else:
        name += "_runall"

common_lines = [
    "#!/bin/bash",
    "#SBATCH --mem=" + args.memory,
    "#SBATCH --time=" + args.time,
    "#SBATCH --partition=bigmem",
    "TMP=$(mktemp -d)",
    "cd /home/tntech.edu/namartinda42/research",
    ". env_setup.sh",
    #". jobs/common.sh",
    ". jobs/common",
    "cd /home/tntech.edu/namartinda42/research/bias"
]

experiment_line = "python experiment.py --experiment ../experiments/" + args.experiment_path + ".json --temp $TMP"
if args.experiment_row is not None:
    experiment_line += " --row " + args.experiment_row
experiment_line += " --log $LOGPATH/" + name + ".log"

common_lines.append(experiment_line)

for i in range(0, len(common_lines)):
    common_lines[i] += "\n"

with open(name + '.sh', 'w') as outfile:
    outfile.writelines(common_lines)

print(name + ".sh")
