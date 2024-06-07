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

...runs the evaluation (Ex.6) on all 3 ZINC datasets ['Train', 'Val', 'Test'] and for all 3 scatter aggregation types ['sum', 'mean', 'max'] once in said orders (9 results total). Alternatively specific datasets and/or scatter-types can be chosen by setting the resp. keywords as stated here as optional arguments; datasets after flag `-d` and scatter-types after flag `-s`, all separated by spaces. E.g. in order to evaluate ZINC_Val & ZINC_Test using scatter_max & scatter_sum, set:

`python main.py -d Val Test -s max sum`

Training is always done on ZINC_Train. The same info can also be found with the `--help` or `-h` flag like so: `python main.py -h`

Since scatter_sum often yielded the best results, it may suffice to run only that like so: `python main.py -s sum`

The remaining exercises 1-5 are covered by the python files in `src`, where each file covers one exercise:

Ex.1: `dataset.py`, Ex.2: `collation.py`, Ex. 3: `layer.py`, Ex.4: `pooling.py`, Ex.5: `virtual_node.py`


## Ex. 6

### Attributes & Parameters

We used the optimizer 'Adam' and the l1-loss function. scatter_sum turned out to be the most promising aggregation type.

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

As mentioned, we likely have yet to find some error of construction somewhere, before we can hope to reach the target MAE.

---

### Note on Exercise Split

In part due to difficult time constraints on both Benedict and Ahmet, David ended up providing most of the codebase (`david/sheet3`) this time around. Benedict greatly helped to further debug and refine the code, and ran a hyperparameter optimization over the parameters mentioned in the list above (HPO not included in `main` yet, see `benedict/sheet3`). Ahmet also made himself available for further improvements on the code.
