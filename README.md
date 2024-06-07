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

If you don't want to have the hassle of fixing all the bugs in current smac hpo:
Just comment out the import statement, and the use of the hpt function. It might be that python does not complain, as long as you don't use the -hpo/--hpt option.
```python
from src.hpo import hpt
# from src.hpo import hpt

...

        ...  = hpt
        # --- = hpt
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

...runs the evaluation (Ex.6) on all 3 ZINC datasets ['Train', 'Val', 'Test'] and for the best of the 3 scatter aggregation types ['sum', 'mean', 'max']. To run other aggregations, scatter-types can be chosen by setting the resp. keywords as stated here as optional arguments;  scatter-types after flag `-s`, all separated by spaces. E.g. in order to evaluate ZINC_Val & ZINC_Test using scatter_max & scatter_sum, set:

`python main.py -s max sum`

Training is always done on ZINC_Train. The same info can also be found with the `--help` or `-h` flag like so: `python main.py -h`

EDIT: Argument parser was changed in Benedict's fork. See `--help`.

Since scatter_sum often yielded the best results, it may suffice to run only that like so: `python main.py -s sum`

The remaining exercises 1-5 are covered by the python files in `src`, where each file covers one exercise:

Ex.1: `dataset.py`, Ex.2: `collation.py`, Ex. 3: `layer.py`, Ex.4: `pooling.py`, Ex.5: `virtual_node.py`


## Ex. 6

Model in `model.py`, parameters in `main.py` or `hpo.py`, respectively.

### Attributes & Parameters

We used the optimizer 'Adam' and the l1-loss function. As expected, scatter_sum turned out to be the most promising aggregation type.

Parameter Values Used:

- Training
    - batch_size (<class 'int'>): 16
    - n_epochs (<class 'int'>): 75
    - weight_decay (<class 'float'>): 2.38002958385e-05
    - use_weight_decay (<class 'numpy.int64'>): 1
    - beta1 (<class 'float'>): 0.9027517490672556
    - beta2 (<class 'numpy.float64'>): 0.999
- GNN
    - scatter_type (<class 'numpy.str_'>): sum
    - n_GNN_layers (<class 'int'>): 5
    - dim_between (<class 'int'>): 32
- Message function (M)
    - n_M_layers (<class 'int'>): 1
    - dim_M (<class 'int'>): 29
    - m_nlin (<class 'numpy.str_'>): leaky_relu
- Update function (U)
    - n_U_layers (<class 'int'>): 3
    - dim_U (<class 'int'>): 30
    - u_nlin (<class 'numpy.str_'>): relu
- Virtual Nodes (VN)
    - use_virtual_nodes (<class 'int'>): 1
    - n_virtual_layers (<class 'int'>): 1
    - (activation fct.: relu)
- (Post-Pooling) MLP
    - n_MLP_layers (<class 'int'>): 1
    - dim_MLP (<class 'int'>): 15
    - mlp_nlin (<class 'numpy.str_'>): relu
- Misc
    - lr (<class 'float'>): 0.0065104040069957
    - lrsched (<class 'numpy.str_'>): cosine
    - use_dropout (<class 'numpy.int64'>): 1
    - dropout_prob (<class 'float'>): 0.4619822213678156
    - use_residual (<class 'numpy.int64'>): 0
    - use_skip (<class 'numpy.int64'>): 1

### Results for Ex. 6

Mean Absolute Error (rounded) on the ZINC datasets, for the chosen scatter operation type:

| Scatter ↓ , Dataset → | Train  | Val    | Test   |
| :-------------------- | :----- | :----- | :----- |
| SUM                   | 0.1167 | 0.3081 | 0.3057 |

## Discussion

We used a BOHB HPO to optimize the hyperparameters, we first had problems achieving low errors, which lay in our bad choice for the dimension-spaces. They were just way to small, when we enlargened this and added more training regularization, error improved drastically.

We still think there might be something wrong in the layer construction, but are not quite sure.

BUT, you can have a look at this beauty:
[wandb.ai/export](https://wandb.ai/gerlach/gnn_zinc/reports/Untitled-Report--Vmlldzo4MjQ5MTAw)

## Conclusion

As mentioned, we maybe have yet to find some error of construction somewhere, and further experiment w/ much higher hidden dimensions for M, U & MLP, before we can hope to reach the target MAE.

---

### Note on Exercise Split

In part due to difficult time constraints on both Benedict and Ahmet, David ended up providing most of the codebase (`david/sheet3`) this time around. Benedict greatly helped to further debug and refine the code, and ran a hyperparameter optimization over the parameters mentioned in the list above. We ended up pushing most of his forked code (`benedict/sheet3`) to the `main` branch. Ahmet also made himself available for further improvements on the code.
