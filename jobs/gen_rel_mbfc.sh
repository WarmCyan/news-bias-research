#!/bin/bash

#SBATCH --mem=64000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 5 --log $LOGPATH/datagen_rel_mbfc_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 6 --log $LOGPATH/datagen_rel_mbfc_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 7 --log $LOGPATH/datagen_rel_mbfc_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 8 --log $LOGPATH/datagen_rel_mbfc_fasttext.log
