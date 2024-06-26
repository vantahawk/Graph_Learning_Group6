# Explanation and Setup for the Code of Group 6 for Sheet 4

(copy from `main` branch)

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

### How to run code for Ex.3 & 4:

```batch
::ON WINDOWS cmdline
...\group6> python src/main.py --task <TASK> --dataset <DATASET>
```

or

```bash
# ON LINUX
.../group6$ python src/main.py --task <TASK> --dataset <DATASET>
```

where `<TASK>` is either `node` or `link` for Ex.3 or 4 respectively, and <br>
where `<DATASET>` is one of `Cora` or `Citeseer` for Ex.3, and  `Facebook` or `PPI` for Ex.4.

By default `node` classification runs for `p = 1.0, q = 0.1`. Other values for `p, q` (as well as for other parameters) can be set independendly for `Cora` & `Citeseer` by (de)commenting their resp. dictionaries in function `main()` in `node_class.py`.

All files in `src` are clearly named accoring to their respective exercises and can be executed independendly to yield sample results.


## Chosen Hyperparameters

### Ex. 3

For Exercise 3, the following hyperparameters were used for each dataset:<br>
We used an HPO to find these.


| Dataset   | sched    | C      | batch_size | delta       | dim  | l   | l_ns | lr       | n_epochs | p   | q   |
|-----------|----------|--------|------------|-------------|------|-----|------|----------|----------|-----|-----|
| Cora      | plateau  | 98.533 | 8726       | 0.005616    | 128  | 5   | 5    | 0.006572 | 250      | 1   | 0.1 |
| CiteSeer  | linear   | 48.541 | 9742       | 0.00001324  | 128  | 5   | 5    | 0.0968   | 200      | 1   | 0.1 |


### Ex. 4

For Exercise 4, the following hyperparameters were used for each dataset:<br>
These were discovered by good intuition after the HPO for task 3.

| Dataset  | sched | C   | batch_size | delta | dim   | l   | l_ns | lr   | n_epochs | p   | q   |
|----------|-------|-----|------------|-------|-------|-----|------|------|----------|-----|-----|
| Facebook | -     | -   | 2000       | -     | 128   | 5   | 5    | 0.01 | 100      | 1.0 | 1.0 |
| PPI      | -     | -   | 2000       | -     | 128   | 5   | 5    | 0.01 | 100      | 1.0 | 1.0 |

The values not given were left at their default values.


## Results

### Ex. 3
__Mean ± StD of Accuracy (rounded in %)__
| Dataset ↓ , p,q → | 1.0, 0.1    | 0.1, 1.0     | 1.0, 1.0     |
| :---------------- | :---------: | :----------: | :----------: |
| Cora              | 85.3 ± 1.93 | 85.78 ± 1.73 | 84.38 ± 2.42 |
| Citeseer          | 63.56 ± 2.0 | 59.78 ± 2.31 | 63.22 ± 2.62 |


### Ex. 4
__Mean ± StD (rounded in %) for p = q = 1.0__
| Dataset  | Accuracy    | ROC-AUC      |
| :------- | :---------: | :----------: |
| Facebook | 97.7 ± 1.29 | 97.78 ± 1.28 |
| PPI      | 86.79 ± 4.4 | 86.8 ± 4.34  |


## Discussion

For Citeseer finding good hyperparameters was difficult, which is why we ran an HPO for that.
For Cora, the results are much better than the requested threshold, whereas for Citeseer, it is relatively tight.

For link prediction, the results are very good, reaching high ROC-AUC scores and accuracies well above the requested thresholds. The hyperparameters were relatively easy to find.

Yet it must be noted that the computation of the tensor of Hadamard products `XX` is rather memory-inefficient and may thus lead to memory issues, e.g. RAM spillovers and associated slowdowns. We tried to fix this issue by finding a more elegant way of mapping/indexing edges to `XX` but have yet to iron things out.

We initially had trouble reaching the desired thresholds until we ran the HPO for Ex.3 and intuited better hyperparameters for Ex.4, e.g. more and much larger batches.

For Ex.3 we again have wandb reports:<br>
[Cora Report](https://api.wandb.ai/links/gerlach/qmpga3sb)
[Citeseer Report](https://api.wandb.ai/links/gerlach/fc1xx2y8)


## Conclusion

This task was more successful than the last ones in achieving the desired results. Accounting for said memory issues, the implementation itself as well as the HPO were relatively fast.

There were however some ambiguities in the exercise: For example whether w & w' in the sum in the denominator of the loss function should be interpreted as sets or sequences, i.e. if they may contain repeated nodes. Moreover all the graphs - except that for `Facebook` - contained connected components with less than two edges - e.g. singular nodes with self-loops - which thus could not satisfy the connectivity conditions set forth for edge sampling in link prediction. At best they might act as redundant edges within the training set. We incdluded an option to remove these invalid connected components from the full edge set beforehand.

Luckily though, none of these issues seemed to lead to much of a performance loss. Likely because (a) the computed loss function values are still similar enough for both the set & sequence interpretation of w & w' in the denominator, and (b) because these too small connected components are negligable compared to the one or few largest ones.

---

### Note on Exercise Split

David laid much of the groundwork for random walks (Ex.1), node2vec (Ex.2), node classification (Ex.3) & link prediction (Ex.4).
Benedict greatly improved upon David's code by making it faster and adding more parallelization, especially for random walks.
He also did the HPO for Ex.3, intuited good hyperparameters for Ex.4 and wrote `main.py`.
Moreover for Ex.4 Benedict improved upon the edge sampling by introducing the building of minimum spanning trees to avoid the removal of connecting edges within train. & eval. edge sets.
Ahmet developed all his code side-by-side, yielding some results of his own. Benedict cleaned and submitted his forked code in the end.


### Later Improvements

On `david/sheet4` the memory issue for the computation of `XX` in link prediction was resolved simply by only taking the Hadamard products over the sampled edges rather than the complete graph. However for some unforseen reason the same issue could not yet be resolved for Benedict's version on `main` while maintaining the previous performance. We can only guess that it has something to do with the different edge sampling strategies used in the versions.

Two more edge sampling strategies for link prediction were optionally added to `david/sheet4`: One which uses minimum spanning trees similar to Benedict's version, and another one which simply re-samples true training edge sets from each connected component until one with a connected subgraph is found. Whereas the tree strategy runs reliably but slow, the re-sampling strategy may not necessarily terminate.
