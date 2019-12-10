#!/bin/bash

#SBATCH --mem=16000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 1 --log $LOGPATH/datagen_fold_0_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 2 --log $LOGPATH/datagen_fold_0_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 3 --log $LOGPATH/datagen_fold_0_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 4 --log $LOGPATH/datagen_fold_0_fasttext.log
