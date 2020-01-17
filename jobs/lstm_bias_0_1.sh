#!/bin/bash
#SBATCH --mem=96000
#SBATCH --time=6:00:00
TMP=$(mktemp -d)
cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common
cd /home/tntech.edu/namartinda42/research/bias
python experiment.py --experiment ../experiments/lstm_bias_0.json --temp $TMP --row 1 --log $LOGPATH/lstm_bias_0_1.log
