#!/bin/bash
TMP=$(mktemp -d)
pushd ../bias
python experiment.py --experiment ../experiments/sentic_tests.json --temp $TMP --row 0 --log ../logs/sentic_tests_0.log
popd