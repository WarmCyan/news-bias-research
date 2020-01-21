#!/bin/bash
TMP=$(mktemp -d)
pushd ../bias
python experiment.py --experiment ../experiments/avg_tests.json --temp $TMP --row 0 --log ../logs/avg_tests_0.log
popd
