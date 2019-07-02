# Learning Fair Naive Bayes Classifiers

## Dependencies
- [Mosek](https://www.mosek.com/downloads/) with a [license](https://www.mosek.com/products/academic-licenses/)
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/en/latest/miniconda.html)
- a c++ compiler

## Setup
To setup the project, checkout to the `master` branch and run:
```
cd setup
./setup.sh
```
## Usage
You should activate the `gpkit-env` environment before using the code. 
```
$ source activate gpkit-env

```

Run the `fair_learn.py` with the `--help` argument to see the usage message:
```
$ python ./fairNB/fair_learn.py --help
usage: fair_learn.py [-h] [--outdir OUTDIR] [--data_dir DATA_DIR]
                     {compas,german,adult} {KLD,Diff} {0.05,0.01,0.1,0.5}
                     {1,10,100}

positional arguments:
  {compas,german,adult}
                        name of the dataset
  {KLD,Diff}            pattern selection heuristic
  {0.05,0.01,0.1,0.5}   fairness threshold
  {1,10,100}            number of patterns

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR
  --data_dir DATA_DIR
```

You can deactivate the environment once you're done using the code
```
$ source deactivate
```

## Installing Mosek
An [installation guide](https://docs.mosek.com/8.1/install/installation.html#general-setup) is provided. 
Following a successfull installation, running `msktestlic` will print a report which ends with this line:
```
************************************
A license was checked out correctly.
************************************
```
