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

### Attributes & Parameters

We used the optimizer 'Adam' and the l1-loss function.

- Training
    - batch size:
    - number of epochs:
- GNN
    - number of GNN layers:
    - dimension between GNN layers:
- Message function (M)
    - number of M layers:
    - hidden dimension of M:
    - activation function of M layers:
- Update function (U)
    - number of U layers:
    - hidden dimension of U:
    - activation function of hidden U layers:
- Virtual Nodes (VN)
    - use virtual nodes: (Yes/No)
    - number of VN-MLP layers:
    - activation function of VN- MLP layers:
- (Post-Pooling) MLP
    - number of MLP layers:
    - hidden dimension of MLP:
    - activation function of hidden MLP layers:

### Results for Ex. 6

Mean Absolute Error (rounded) on the ZINC datasets, for each scatter operation type:

| Scatter ↓ , Dataset → | Train      | Val        | Test       |
| :-------------------- | :--------- | :--------- | :--------- |
| SUM                   |  |  |  |
| MEAN                  |  |  |  |
| MAX                   |  |  |  |

## Discussion



## Conclusion



---

### Note on Exercise Split
