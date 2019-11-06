#!/bin/bash
#SBATCH --mem=96000
#SBATCH --time=6:00:00
cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias
python experiment.py --experiment ../experiments/cnn_reliability_0.json --row 10 --log $LOGPATH/cnn_reliability_0_10.log
