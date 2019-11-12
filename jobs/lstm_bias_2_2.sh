#!/bin/bash
#SBATCH --mem=64000
#SBATCH --time=6:00:00
TMP=$(mktemp -d)
cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias
python experiment.py --experiment ../experiments/lstm_bias_2.json --temp $TMP --row 2 --log $LOGPATH/lstm_bias_2_2.log
