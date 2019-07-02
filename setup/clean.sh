#!/bin/bash

source deactivate
conda remove --yes --name gpkit-env --all

cd "$(dirname "$0")"/..
rm -rf fairNB/build
rm -rf fairNB/pattern_finder/pf_wrapper.cpp
rm -f fairNB/pattern_finder/pf_src/build
rm -rf fairNB/pattern_finder/*.so    
