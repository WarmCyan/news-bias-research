#!/bin/bash

#SBATCH --mem=16000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 13 --log $LOGPATH/datagen_fold_3_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 14 --log $LOGPATH/datagen_fold_3_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 15 --log $LOGPATH/datagen_fold_3_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 16 --log $LOGPATH/datagen_fold_3_fasttext.log
