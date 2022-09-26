# replay_identification

[![PR Test](https://github.com/Eden-Kramer-Lab/replay_identification/actions/workflows/PR-test.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/replay_identification/actions/workflows/PR-test.yml)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/replay_identification/python-version)
[![DOI](https://zenodo.org/badge/105480682.svg)](https://zenodo.org/badge/latestdoi/105480682)

A semi-latent state-space model that combines movement, LFP, and ensemble single unit/multiunit information to identify periods of replay and decode its content.

NOTE: This code is still in production and prepublication.

### Installation

`replay_identification` can be installed through pypi or conda. Conda is the best way to ensure that everything is installed properly.

```bash
pip install replay_identification
python setup.py install
```

Or

```bash
conda install -c edeno replay_identification
python setup.py install
```

### Usage

See the notebooks ([\#1](https://nbviewer.jupyter.org/github/Eden-Kramer-Lab/replay_identification/blob/master/notebooks/Test_Simulated_Data.ipynb), [\#2](https://nbviewer.jupyter.org/github/Eden-Kramer-Lab/replay_classification/blob/master/examples/Test_Real_Data.ipynb)) for more information on how to use the package.

You can also use the `launch binder` button at the top of the Readme to play with simulated data in your web browser.

### Package Requirements ###

+ numpy
+ scipy
+ statsmodels
+ numba
+ matplotlib
+ xarray
+ scikit-learn
+ regularized_glm

See the `setup.py`or `environment.yml` file for the most up to date list of dependencies.

### Developer Installation ###

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Go to the local repository (`.../replay_identification`) and install the anaconda environment for the repository. Type into bash:

```bash
conda update -q conda
conda info -a
conda env create -f environment.yml
source activate replay_identification
python setup.py develop
```
