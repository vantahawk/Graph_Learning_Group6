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

### How to run code for Ex.3 or Ex.4:

```batch
::ON WINDOWS cmdline
...\group6> python src/main.py --task <TASK> --dataset <DATASET>
```

or

```bash
# ON LINUX
.../group6$ python src/main.py --task <TASK> --dataset <DATASET>
```

where `<TASK>` is either `node` or `link` for Exercise 3 and 4, respectively, and <br>
where `<DATASET>` is one of `Cora`, `Citeseer` for Ex.3, and  `Facebook`, `PPI` for Ex. 4.


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

Dataset | Accuracy in %
---: | :---:
Cora | 85.3 ± 1.93
CiteSeer | 63.56 ± 2.0

### Ex. 4

Dataset | Accuracy in % | ROC-AUC in %
---: | :---: | :---:
Facebook | 97.7 ± 1.29 |97.78 ± 1.28
PPI | 86.79 ± 4.4 | 86.8 ± 4.34

## Discussion

For Citeseer finding good hyperparameters was difficult. Thats why we made the HPO.
For Cora, the results are much better than the requested 0.75. But for Citeseer, it's relatively tight.

For link prediction, the results are very good. The ROC-AUC is very high, and the accuracy is also very good. And the hyperparameters were relatively easy to find.

## Conclusion

This task was more successful than the last ones, in achieving the desired values.
Implementation was quite fast, HPO was fast. 
Overall good entire exercise.

### Note on Exercise Split
David started on the node2vev and node classification and implemented most things roughly, but quite well.
Same goes for the link prediction.
Benedict improved upon his code by making it faster, and more parallelized. He did the hpo for the ex.3.
For ex.4 he improved upon the edge sampling by introducing the building of the spanning tree, to remove necessary edges.
Ahmet developed all his code side-by-side. He and David had trouble achieving the desired accuracies, probably because they choose the hyperparameters not well. 
Benedict cleaned the submitted code in the end.

