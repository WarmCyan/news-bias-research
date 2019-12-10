#!/bin/bash

#SBATCH --mem=16000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 33 --log $LOGPATH/datagen_fold_8_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 34 --log $LOGPATH/datagen_fold_8_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 35 --log $LOGPATH/datagen_fold_8_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 36 --log $LOGPATH/datagen_fold_8_fasttext.log
