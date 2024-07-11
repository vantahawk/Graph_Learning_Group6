# Explanation and Setup for the Code of Group 6 for Sheet 3

## Requirements

Use Python 3.12, other versions have not been tested and are thus not necessarily working.

Install the requirements either in a conda environment or in a virtualenv. The latter can be done like this:

BASH

```bash
#ON LINUX
.../group6$ python -m venv venv
.../group6$ venv/bin/activate
.../group6$ python -m pip install -r requirements.txt
```

BATCH/CMDLine

```batch
::ON WINDOWS cmdline, NOT powershell
...\group6> python -m venv venv
...\group6> .\venv\Scripts\activate.bat
:: in powershell use `.\venv\Scripts\Activate.ps1` instead
...\group6> python -m pip install -r requirements.txt
```

You further must install torch_scatter in the appropriate version for your system. This can be done by running the following command from the requirements.txt.


## How to run

```batch
::ON WINDOWS cmdline
...\group6> python main.py
```

or

```bash
# ON LINUX
.../group6$ python main.py
```

runs 5-fold CV and saves the best predictions at the end.


## Setup

This code is mainly based on Exercise 3. But some features were engineered.

This can all be found in `src/dataset.py`.

### Electronical Features

For more information based on real physical values, we provided the model with eletronical features, per atom.
These are:
 - atomic number <- based on the node_label
 - number of valence electrons
 - number of bonds
 - electro negativity 
 - electro affinity (zero-energy)
 - ionization energies up to the 4th ionization

### Kernels
For these some properties can be changed in `src/feature_config.yaml`
1. hosoya index
        The hosoya index is often used in chemical graph theory.
        We therefore gave each node the complete hosoya index of the graph, as a feature.
        Further 5 samples of a node-centered subgraph hosoya index to capture local structure.
2. closed walk
        The closed walk kernel was changed in that each node gets only the number of cycles its part of. This is important for rings in molecules.

### Distances
We used distances as node and edge features. For the latter obviously only the bond lenght, but for the node features all the distances to the other atoms. as this influences stuff like van der waals forces.

### Other
The molecule was further translated into a zmatrix based on the given 3d coordinates. This was done to get a better understanding of the molecule and its structure.
This is based on some tree through the graph, it may help to include multiple zmatrices, but we did not.

For edge features we also incorporated the given bond types.

### Notes

We generally did not normalize the features, except for some small normalization for some, because it must be done globally over all molecules, otherwise the model would not be able to learn the differences between the molecules correctly. We were dumb at this, could have been done better.

## Hyperparameters

Can be seen in the main.py file. We used optuna for optimizing those.


## Results
Our validation loss is around 0.3, which is relatively good, but other model types may have yielded better ones. But the molecules were quite small, so maybe not.
For predicting the final test set, the validation set is added to the test set if the early stopping kicks in or we ran through the full epochs. Then another 20 epochs are trained.

The saved predictions are from the model that had the best validation score before restart.
