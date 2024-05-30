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

## How to run

How to run code for Ex.6:

```batch
::ON WINDOWS cmdline
...\group6> python main.py
```

or

```bash
# ON LINUX
.../group6$ python main.py
```

(describe what main.py does with datasets, command line arg.s & --help, repo structure & notes on Ex.1-5, etc.)

## Ex. 6

### Model Attributes & Parameters

We used the optimizer 'Adam' and the l1-loss function.

- batch size for ZINC_Train:
- batch size for ZINC_Val:
- batch size for ZINC_Test:
- number of epochs for ZINC_Train:
- number of epochs for ZINC_Val:
- number of epochs for ZINC_Test:

### Results for Ex. 6

Mean Absolute Error (rounded in %) on the ZINC datasets:

| Dataset | Result       |
| :------ | :----------- |
|Train    |  |
|Val      |  |
|Test     |  |

## Discussion



## Conclusion



---

### Note on Exercise Split
