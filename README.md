# begrs_varma

This repository contains the replication files for the BEGRS application on VARMA models

## Requirements and Installation

Running replication files require:
- The `begrs` toolbox with dependencies installed.
- Additional packages specified in `requirements_extra.txt`

Note: the files were run using GPU-enabled HPC nodes, therefore any attempt at replication should take into account this computational requirement. This is particularly the case for the SBC analysis, which is time-consuming even on an HPC node. The files are provided for the sake of transparency and replication, and all results are provided in the associated release (see below).

## Release contents

The release provides zipped versions of the following folders. These contain all the intermediate results of the scripts, so that the outputs of the paper (i.e. figures) can be generated directly from them, without requiring a full re-run of the entire analysis.

- `/abc_smc`: contains the results of the ABC-SMC estimations.
- `figures`: contains the figures used in the paper
- `models`: contains the saved trained BEGRS surrogate models and their associated posterior estimates
- `sbc`: contains the results of the SBC diagnostic

## Run sequence:

The various scripts should be run in the following order, as the outputs of earlier scripts for the inputs of later ones. To run a later file (e.g. output generation) without running an earlier file (e.g. estimation), use the folders provided in the release as the source of the intermediate inputs. Files within a subsection can be run in any order.

### 1. Generate simulation data

- `varma_sim.py` - Generate VAR/VARMA simulation data (training + testing)

### 2. Run estimations

 Run the ABC-SMC estimation, the BEGRS estimation and the SBC diagnostic on the training and testing data.

- `varma_abc_est.py` - Estimate the VAR/VARMA model using ABC-SMC from one run of testing data
- `begrs_varma_train.py` - Train a begrs object on the simulated VAR/VARMA data
- `begrs_varma_est.py` - Estimate the VAR/VARMA model using BEGRS from one run of testing data
- `begrs_varma_sbc.py` - Run a SBC diagnostic on the full set of testing data

### 3. Generate outputs

- `begrs_varma_est_outputs.py` - Generate outputs from BEGRS and ABC-SMC estimations for the paper
- `begrs_varma_sbc_outputs.py` - Generate outputs from SBC diagnostic of BEGRS for the paper
