import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="name", default=None, required=True, help="name")
parser.add_argument("-e", dest="experiment_path", default=None, help="experiment file name", required=True)
parser.add_argument("-r", dest="experiment_row", default=None, help="row")
parser.add_argument("-m", dest="memory", default="32000", help="memory (32000)")
parser.add_argument("-t", dest="time", default="1:00:00", help="time (1:00:00)")
args = parser.parse_args()

common_lines = [
    "#!/bin/bash",
    "#SBATCH --mem=" + args.memory,
    "#SBATCH --time=" + args.time,
    "cd /home/tntech.edu/namartinda42/research",
    ". env_setup.sh",
    ". jobs/common.sh",
    "cd /home/tntech.edu/namartinda42/research/bias"
]

experiment_line = "python experiment.py --experiment ../experiments/" + args.experiment_path
if args.experiment_row is not None:
    experiment_line += " --row " + args.experiment_row
experiment_line += " --log $LOGPATH/" + args.name + ".log"

common_lines.append(experiment_line)

for i in range(0, len(common_lines)):
    common_lines[i] += "\n"

with open(args.name + '.sh', 'w') as outfile:
    outfile.writelines(common_lines)

print(args.name + ".sh")
