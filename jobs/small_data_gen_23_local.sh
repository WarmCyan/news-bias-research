#!/bin/bash
TMP=$(mktemp -d)
pushd ../bias
python experiment.py --experiment ../experiments/small_data_gen.json --temp $TMP --row 23 --log ../logs/small_data_gen_23.log
popd
