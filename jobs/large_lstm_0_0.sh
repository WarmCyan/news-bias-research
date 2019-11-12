#!/bin/bash
#SBATCH --mem=96000
#SBATCH --time=6:00:00
TMP=$(mktemp -d)
cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias
python experiment.py --experiment ../experiments/large_lstm_0.json --temp $TMP --row 0 --log $LOGPATH/large_lstm_0_0.log
