#!/bin/bash
TMP=$(mktemp -d)
pushd ../bias
python experiment.py --experiment ../experiments/small_data_gen.json --temp $TMP --row 16 --log ../logs/small_data_gen_16.log
popd
