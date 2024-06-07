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

Parameter Values Used:
 - weight_decay (<class 'float'>): 2.38002958385e-05
 - use_virtual_nodes (<class 'int'>): 1
 - n_virtual_layers (<class 'int'>): 1
 - dim_U (<class 'int'>): 30
 - batch_size (<class 'int'>): 16
 - beta1 (<class 'float'>): 0.9027517490672556
 - beta2 (<class 'numpy.float64'>): 0.999
 - dim_M (<class 'int'>): 29
 - dim_MLP (<class 'int'>): 15
 - dim_between (<class 'int'>): 32
 - lr (<class 'float'>): 0.0065104040069957
 - lrsched (<class 'numpy.str_'>): cosine
 - m_nlin (<class 'numpy.str_'>): leaky_relu
 - mlp_nlin (<class 'numpy.str_'>): relu
 - n_GNN_layers (<class 'int'>): 5
 - n_MLP_layers (<class 'int'>): 1
 - n_M_layers (<class 'int'>): 1
 - n_U_layers (<class 'int'>): 3
 - n_epochs (<class 'int'>): 75
 - scatter_type (<class 'numpy.str_'>): sum
 - u_nlin (<class 'numpy.str_'>): relu
 - use_dropout (<class 'numpy.int64'>): 1
 - use_residual (<class 'numpy.int64'>): 0
 - use_skip (<class 'numpy.int64'>): 1
 - use_weight_decay (<class 'numpy.int64'>): 1
 - dropout_prob (<class 'float'>): 0.4619822213678156

### Results for Ex. 6

Mean Absolute Error (rounded) on the ZINC datasets, for the chosen scatter operation type:

Scatter ↓
sum:	 (0.1166774183511734, 0.30809709429740906, 0.3057083189487457)


| Scatter ↓ , Dataset → | Train      | Val        | Test       |
| :-------------------- | :--------- | :--------- | :--------- |
| SUM                   | 0.1166774183511734 | 0.30809709429740906 | 0.3057083189487457 |

## Discussion
We used a BOHB HPO to optimize the hyperparameters, we firsdt had problems achieving low errors, which lay in our bad choice for the dimension-spaces. They were just way to small.


## Conclusion



---

### Note on Exercise Split
