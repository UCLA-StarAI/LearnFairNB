#!/bin/bash

cd "$(dirname "$0")"/..
rm -rf fairNB/build
rm -rf fairNB/pattern_finder/pf_wrapper.cpp
rm -f fairNB/pattern_finder/pf_src/build
rm -rf fairNB/pattern_finder/*.so  

source deactivate
source activate gpkit-env 

cd fairNB 
python setup.py build_ext --inplace 
echo -e "\n\nTesting the pattern finder:\n" 
python pattern_finder/pattern_finder.py 

source deactivate 
