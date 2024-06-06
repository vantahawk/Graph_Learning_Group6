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
...runs the evaluation (Ex.6) on all 3 ZINC datasets ['Train', 'Val', 'Test'] and for all 3 scatter aggregation types ['sum', 'mean', 'max'] once in said orders (9 results total). Alternatively specific datasets and/or scatter-types can be chosen by setting the resp. keywords as stated here as optional arguments; datasets after flag `-d` and scatter-types after flag `-s`, all separated by spaces. E.g. in order to evaluate ZINC_Val & ZINC_Test using scatter_max & scatter_sum, set:

`python main.py -d Val Test -s max sum`

Training is always done on ZINC_Train. The same info can also be found with the `--help` or `-h` flag like so: `python main.py -h`

Since scatter_sum usually yielded the best results, as it did in the table below, it may suffice to run only that like so: `python main.py -s sum`  (# TODO see if true)

The remaining exercises 1-5 are covered by the python files in `src`, where each file covers one exercise:

Ex.1: `dataset.py`, Ex.2: `collation.py`, Ex. 3: `layer.py`, Ex.4: `pooling.py`, Ex.5: `virtual_node.py`


## Ex. 6

### Attributes & Parameters

We used the optimizer 'Adam' and the l1-loss function. (scatter_sum turned out to be the most promising aggregation type. (# TODO see if true))

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

| Scatter ↓ , Dataset → | Train | Val  | Test |
| :-------------------- | :---  | :--- | :--- |
| SUM                   |       |      |      |
| MEAN                  |       |      |      |
| MAX                   |       |      |      |


## Discussion



## Conclusion



---

### Note on Exercise Split

In part due to difficult time constraints on both Benedict and Ahmet, David ended up providing most of the codebase this time around. Benedict greatly helped to further debug and refine the code, and ran a (limited) hyperparameter optimization over the parameters mentioned in the list above. Ahmet also made himself available for further improvements on the code.
