#!/bin/bash
	
source deactivate
conda remove --yes --name gpkit-env --all

conda create --yes --name gpkit-env python=2.7 pandas cython

source activate gpkit-env
pip install timeout-decorator

cd "$(dirname "$0")"/..
pip install -e gpkit
cd  .. 
python  -c  "import  gpkit.tests;  gpkit.tests.run()"
source deactivate 