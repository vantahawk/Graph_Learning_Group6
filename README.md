# Explanation and Setup for the Code of Group6 for Sheet 2

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

How to run code for Ex.5 & 6:

```batch
::ON WINDOWS cmdline
...\group6> python main.py
```

or

```bash
# ON LINUX
.../group6$ python main.py
```

...runs GCN model training & testing resp. for a number of epochs for all 4 datasets 'ENZYMES', 'NCI1', 'Citeseer' & 'Cora' once in said order on the respective model type, i.e. graph-level (Ex.5) for ENZYMES & NCI1 and nonde-level (Ex.6) for Citeseer & Cora.

Alternatively specific datasets can be chosen by setting the resp. keywords 'enzymes', 'nci1', 'citeseer' or 'cora' as positional arguments. For example to run NCI1 & Cora in said order set:

`python main.py nci1 cora`

The same info can also be found with the `--help` or `-h` flag like so: `python main.py -h`

The preprocessing functions (Ex.1) and the single GCN layer class (Ex.2) can be found in files `src/preprocessing.py` and `src/layers.py` resp. The full model classes for graph- (Ex.3) & node-level (Ex.4) can be found in `src/models.py`. `main.py` also contains all the functions necessary for the transformation, segmentation and formatting of the data.

## Ex. 5 & 6

### Model Attributes & Parameters

We used the optimizer 'Adam' and the cross entroy loss function.

- Number of splits for cross-validation (Ex.5) or number of rounds (Ex.6): 10 (as discussed in exercise)
- batch size for ENZYMES: 100
- batch size for NCI1: 100
- number of epochs forENZYMES: 20
- number of epochs for NCI1: 20
- number of epochs for Citeseer: 150
- number of epochs for Cora: 150

### Results for Ex. 5 & 6

Mean ± Standard Deviation of Accuracy Scores (rounded in %):

| :------ | :------------ |
|ENZYMES  | 15.17  ± 11.58 |
|NCI1     | 65.82  ± 5.77 |
|Citeseer | 67.92  ± 0.98 |
|Cora     | 67.69  ± 0.62 |

## Discussion

(outline, in progress)
- unfortunately did not reach any of the accuracy targets in the exercise sheet
- got at least close-ish for Citeseer & Cora
- might need to experiment with dropout layers
- loss consistently decreases but not very rapidly
- accuracies on training data perform well, consistently rising up to 80-90+%
- accuracies for ENZYMES behave normally on training data but vary wildy on test data, randomly dropping to zero, hence the poor end result, possible error somewhere during testing phase, still unknown as of yet
- accuracies for test data of NCI1 are also somewhat unstable, varying between ~40-70% w/o clear direction, possibly due to overfitting, likely not though since there is no clear decrease over epochs

## Conclusion

(outline, in progress)
- if the issue with NCI1 is indeed due to overfitting, experimenting more with dropout layers might help
- issue with ENZYMES still needs to be resolved
- experimenting more with parameters like number of epochs and batch size, as well as hyperparameters of the optimizer, might yield further improvements for Citeseer & Cora, likely less so for ENZYMES & NCI1 though
