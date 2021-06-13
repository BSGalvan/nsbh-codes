# nsbh-codes
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## NSBH Mergers and their EM Counterparts

This repo houses codes written for and to accompany my Master's thesis titled
"Multimessenger outflows from neutron star mergers."

## Quick Start

In order to test these codes out on your local machine:

- Download and install
  [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- You will then need the IGWN conda environment. The environment YAML files can
  be found [here](https://computing.docs.ligo.org/conda/environments/). These
  codes were written with the `igwn-py38` environment, although past/future
  environments shouldn't break these codes (if they do, report it in under
  Issues!)
- Once you have both miniconda and the IGWN environment file, change to the
  directory in which the YAML file is located and create the environment using
  `conda env create --file igwn-py38.yaml`
- Clone the repo : `git clone https://github.com/BSGalvan/nsbh-codes.git`
- Test populations and SNRs calculated for those populations are stored under
  `data/`, so you can directly use those. Simply change the variables pointing
  to them inside `explore_pop.py`


## TODO

- [ ] command line arguments for user-specific population and SNR files
- [ ]
- [x] wrap ECDF + median + plotting code into separate plots module
- ~~[ ]~~ commandline args for `create_pop.py` and `explore_pop.py`
  Edit: deemed unnecessary
- ~~[ ]~~ fix `gen_samples.py` to reflect new (more rigorous) ecdf calculation
  Edit: the current implementation is rigorous for that purpose
- ~~[ ]~~ finish `em_signals.py` to compute KN lightcurves
  Edit: deferred for the future
