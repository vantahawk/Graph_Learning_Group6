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

### How to run code for Ex.3:

```batch
::ON WINDOWS cmdline
...\group6> python src/node_class.py --dataset <DATASET>
```

or

```bash
# ON LINUX
.../group6$ python src/node_class.py --dataset <DATASET>
```

where `<DATASET>` is one of `Cora`, `Citeseer`.

### How to run code for Ex.4:

```batch
::ON WINDOWS cmdline
...\group6> python src/link_pred.py --dataset <DATASET>
```

or

```bash
# ON LINUX
.../group6$ python src/link_pred.py --dataset <DATASET>
```

where `<DATASET>` is one of `Facebook'`, `PPI`.


## Chosen Hyperparameters

### Ex. 3

For Exercise 3, the following hyperparameters were used for each dataset:<br>
We used an HPO to find these.


| Dataset   | sched    | C      | batch_size | delta       | dim  | l   | l_ns | lr     | n_epochs | p   | q   |
|-----------|----------|--------|------------|-------------|------|-----|------|--------|----------|-----|-----|
| Cora      | plateau  | 98.533 | 8726       | 0.005616    | 128  | 5   | 5    | 0.006572 | 250      | 1   | 0.1 |
| CiteSeer  | linear   | 48.541 | 9742       | 0.00001324  | 128  | 5   | 5    | 0.0968   | 200      | 1   | 0.1 |


### Ex. 4

For Exercise 4, the following hyperparameters were used for each dataset:<br>
These were discovered by good intuition after the HPO for task 3. 

Dataset | sched | C | batch_size | delta | dim | l | l_ns | lr | n_epochs | p | q
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Facebook | - | - | 2000 | - | 128 | 5 | 5 | 0.01 | 100 | 1.0 | 1.0
PPI | - | - | 2000 | - | 128 | 5 | 5 | 0.01 | 100 | 1.0 | 1.0

the non-given values were left at their default values.

## Results

### Ex. 3

Dataset | Accuracy
---: | :---:
Cora | 0.83 ± 0.01
CiteSeer | 0.72 ± 0.01

### Ex. 4

Dataset | Accuracy | ROC-AUC
---: | :---: | :---:
Facebook | 97.7 ± 1.29 |97.78 ± 1.28
PPI | 86.79 ± 4.4 | 86.8 ± 4.34

## Discussion

For Citeseer finding good hyperparameters was difficult. Thats why we made the HPO.
For Cora, the results are much better than the requested 0.75. But for Citeseer, it's relatively tight.

For link prediction, the results are very good. The ROC-AUC is very high, and the accuracy is also very good. And the hyperparameters were relatively easy to find.

## Conclusion

As mentioned, we maybe have yet to find some error of construction somewhere, and further experiment w/ much higher hidden dimensions for M, U & MLP, before we can hope to reach the target MAE.

### Note on Exercise Split


