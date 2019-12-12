#!/bin/bash
#SBATCH --mem=96000
#SBATCH --time=6:00:00
TMP=$(mktemp -d)
cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias
python experiment.py --experiment ../experiments/lstm_bias_0.json --temp $TMP --row 6 --log $LOGPATH/lstm_bias_0_6.log
