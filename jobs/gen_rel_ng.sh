#!/bin/bash

#SBATCH --mem=64000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 9 --log $LOGPATH/datagen_rel_ng_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 10 --log $LOGPATH/datagen_rel_ng_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 11 --log $LOGPATH/datagen_rel_ng_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 12 --log $LOGPATH/datagen_rel_ng_fasttext.log
