#!/bin/bash

#SBATCH --mem=16000
#SBATCH --time=2:00:00

cd /home/tntech.edu/namartinda42/research
. env_setup.sh
. jobs/common.sh
cd /home/tntech.edu/namartinda42/research/bias

python experiment.py --experiment ../experiments/data_gen.json --row 17 --log $LOGPATH/datagen_bias_as_tfidf.log
python experiment.py --experiment ../experiments/data_gen.json --row 18 --log $LOGPATH/datagen_bias_as_w2v.log
python experiment.py --experiment ../experiments/data_gen.json --row 19 --log $LOGPATH/datagen_bias_as_glove.log
python experiment.py --experiment ../experiments/data_gen.json --row 20 --log $LOGPATH/datagen_bias_as_fasttext.log
