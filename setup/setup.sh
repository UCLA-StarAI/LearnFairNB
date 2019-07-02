#!/bin/bash

cd "$(dirname "$0")"

./clean.sh
./setup_environment.sh
./setup_finder.sh

# running a test 
source activate gpkit-env
mkdir -p test_output
python ../fairNB/fair_learn.py compas KLD 0.1 10 --outdir ./test_output
source deactivate
