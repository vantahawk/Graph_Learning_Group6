# Explanation and Setup for the Code of Group6 for Sheet 2

## Requirements

Use Python 3.12, other versions have not been tested and are thus not necessarily working.

Install the requirements either in a conda environment or in a virtualenv. The latter can be done like this:

---

BASH

```bash
#ON LINUX
.../group6$ python -m venv venv
.../group6$ venv/bin/activate
.../group6$ python -m pip install -r requirements.txt
```

---

BATCH/CMDLine

```batch
::ON WINDOWS cmdline, NOT powershell
...\group6> python -m venv venv
...\group6> .\venv\Scripts\activate.bat
:: in powershell use `.\venv\Scripts\Activate.ps1` instead
...\group6> python -m pip install -r requirements.txt
```

Then go into the site-packages folder and clone [this repo](https://github.com/automl/ConfigSpace.git), switch to branch `typing-cython-3` and install it with `python -m pip install -e .`.
This will install the ConfigSpace package as .egg from the git locally.

If you don't want to do that, comment out all parts relevant to the hpo in the code, most of which is in `src/utils.py`.
Only the imports and use in `main.py` also need to be commented out.
In `src/utils.py` make sure you let the `torchModel` class survive, because it is used in the main code, only comment out the configspace and smac stuff, e.g. also the warnings module.

## How to run


To run code for the exercises, see the help page:

```batch
::ON WINDOWS cmdline
...\group6> python src/main.py --help
```

```bash
# ON LINUX
.../group6$ python src/main.py --help
```

This will show you the available options.

If you want to run the code without optimizing hyperparameters, but using the default ones, set the appropriate flag to True.
This is highly recommended, as optimization takes a long time. If you want to mess with that, feel free to tune the bounds on the number of iterations and the wallclock time [right here](src/utils.py#365).

Your normal command will look something like this:

```bash
# ON LINUX
.../group6$ python src/main.py --l node --d cuda --ds Cora --def-hps True
```

```batch
::ON WINDOWS cmdline
...\group6> python src/main.py --l node --d cuda --ds Cora --def-hps True
```

## Achieved Accuracies

Dataset | Cora  | Citeseer | ENZYME | NCI1
--- | --- | --- | --- | ---
MEAN | 0.630 | 0.548 | 0.615 | 0.779
STD | 0.016 | 0.006 | 0.061 | 0.027

## Notes

We did split the work initially, but since the work could only really be split into 2 (Node/Graph-level classification), we did it like this, but the entire code comes from Benedict, see the individual branches for the work of the others.

We noticed that data normalization helped extremely with results.
The hpo was probably a bit overkill, but it did improve our results slightly and we never let if fully run, as its really time-consuming, even though its a BOHB implementation, so maybe it could have helped more.

## Repo Structure

We had three branches for each group member, the code structure was thrown together preliminarily by Benedict, in the beginning, so that it would be easier to merge later.

1. main.py - the main entry point, also hosts the main train/eval loop for the experiments
2. utils.py - various other code, related to data, hpo, and other utilities
3. models.py - the models used for the experiments
4. layers.py - the layers used for the models
5. preprocessing.py - the preprocessing done on the data (as described in exercise 1)
6. decorators.py - the parseargs decorator to parse arguments beautifully into any function, wrapping the whole messy argparse thing

