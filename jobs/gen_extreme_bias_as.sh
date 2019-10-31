#!/bin/bash

#SBATCH --mem=32000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 25 --log $LOGPATH/datagen_extreme_bias_as_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 26 --log $LOGPATH/datagen_extreme_bias_as_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 27 --log $LOGPATH/datagen_extreme_bias_as_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 28 --log $LOGPATH/datagen_extreme_bias_as_fasttext.log
