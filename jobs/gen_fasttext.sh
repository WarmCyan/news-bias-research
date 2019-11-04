#!/bin/bash

#SBATCH --mem=64000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 4 --log $LOGPATH/fasttext4.log
python experiment.py --experiment ../experiments/data_gen.json --row 8 --log $LOGPATH/fasttext8.log
python experiment.py --experiment ../experiments/data_gen.json --row 12 --log $LOGPATH/fasttext12.log
python experiment.py --experiment ../experiments/data_gen.json --row 16 --log $LOGPATH/fasttext16.log
python experiment.py --experiment ../experiments/data_gen.json --row 20 --log $LOGPATH/fasttext20.log
python experiment.py --experiment ../experiments/data_gen.json --row 24 --log $LOGPATH/fasttext24.log
python experiment.py --experiment ../experiments/data_gen.json --row 28 --log $LOGPATH/fasttext28.log
