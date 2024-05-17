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
.../group6$ python src/main.py --l node --d cuda --ds Cora
```

```batch
::ON WINDOWS cmdline
...\group6> python src/main.py --l node --d cuda --ds Cora
```

## Achieved Results

Dataset | Cora  | Citeseer | ENZYME | NCI1
--- | --- | --- | --- | ---
MEAN | 
STD | 

## Notes

We did split the work initially, but since the work could only really be split into 2 (Node/Graph-level classification), we did it like this, but the entire code comes from Benedict, see the individual branches for the work of the others.

