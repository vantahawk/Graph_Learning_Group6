# Explanation and Setup for the Code of Group 6 for Sheet 4

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

How to run code for Ex.3 & 4:

```batch
::ON WINDOWS cmdline
...\group6> python main.py
```

or

```bash
# ON LINUX
.../group6$ python main.py
```

TODO (update & finish readme)

...runs the evaluation (Ex.6) on all 3 ZINC datasets ['Train', 'Val', 'Test'] and for all 3 scatter aggregation types ['sum', 'mean', 'max'] once in said orders (9 results total). Alternatively specific datasets and/or scatter-types can be chosen by setting the resp. keywords as stated here as optional arguments; datasets after flag `-d` and scatter-types after flag `-s`, all separated by spaces. E.g. in order to evaluate ZINC_Val & ZINC_Test using scatter_max & scatter_sum, set:

`python main.py -d Val Test -s max sum`

Training is always done on ZINC_Train. The same info can also be found with the `--help` or `-h` flag like so: `python main.py -h`

Since scatter_sum often yielded the best results, it may suffice to run only that like so: `python main.py -s sum`

The remaining exercises 1-5 are covered by the python files in `src`, where each file covers one exercise:

Ex.1: `dataset.py`, Ex.2: `collation.py`, Ex. 3: `layer.py`, Ex.4: `pooling.py`, Ex.5: `virtual_node.py`


## Ex. 6

### Attributes & Parameters

We used the optimizer 'Adam' and the l1-loss function. scatter_sum turned out to be the most promising aggregation type.

- Training
    - batch size: 10
    - number of epochs: 20
- GNN
    - number of GNN layers: 2
    - dimension between GNN layers: 5
- Message function (M)
    - number of M layers: 1
    - hidden dimension of M: 5
    - activation function of M layers: ReLU
- Update function (U)
    - number of U layers: 2
    - hidden dimension of U: 5
    - activation function of hidden U layers: ReLU
- Virtual Nodes (VN)
    - use virtual nodes: Yes
    - number of VN-MLP layers: 1
    - activation function of VN- MLP layers: ReLU
- (Post-Pooling) MLP
    - number of MLP layers: 2
    - hidden dimension of MLP: 5
    - activation function of hidden MLP layers: ReLU

### Results for Ex. 6

Mean Absolute Error (rounded) on the ZINC datasets, for each scatter aggregation type:

| Scatter ↓ , Dataset → | Train | Val  | Test |
| :-------------------- | :---- | :--- | :--- |
| SUM                   |  0.49 | 0.49 | 0.54 |
| MEAN                  |  0.56 | 0.57 | 0.62 |
| MAX                   |  0.55 | 0.57 | 0.62 |


## Discussion

The mean absolute error (MAE) develops generally as expected, with MAEs decaying roughly asymptotically for all 3 datasets with increasing epochs; usually most for ZINC_Train, less so for ZINC_Val and least for ZINC_Test.

Unfortunately though we did not manage to reach the target MAE of 0.2 for ZINC_Test.

Generally more than 2-5 GNN layers did not yield notably better or yielded worse results and only extended computation time unnecessarily. Performance and computation time also tend to worsen for a number of epochs below or high above a magnitude of around 10.

Errors within dataset.py, collation.y or layer.py were considered as reasons for the subpar performance, but none were found and we figured, if present, they would yield much worse results even. There might be issues in how the layer module (lists) in GNN_Layer are constructed, but we also found no further error in there.


## Conclusion

As mentioned, we likely have yet to find some error of construction somewhere, before we can hope to reach the target MAE.

---

### Note on Exercise Split

In part due to difficult time constraints on both Benedict and Ahmet, David ended up providing most of the codebase (`david/sheet3`) this time around. Benedict greatly helped to further debug and refine the code, and ran a hyperparameter optimization over the parameters mentioned in the list above. Ahmet also made himself available for further improvements on the code.
