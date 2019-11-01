#!/bin/bash
#SBATCH --mem=16000
#SBATCH --time=2:00:00
cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias
python experiment.py --experiment ../experiments/datagen.json --row 1 --log $LOGPATH/testing
