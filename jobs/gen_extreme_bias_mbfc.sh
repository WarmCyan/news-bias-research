#!/bin/bash

#SBATCH --mem=16000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 21 --log $LOGPATH/datagen_extreme_bias_mbfc_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 22 --log $LOGPATH/datagen_extreme_bias_mbfc_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 23 --log $LOGPATH/datagen_extreme_bias_mbfc_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 24 --log $LOGPATH/datagen_extreme_bias_mbfc_fasttext.log
