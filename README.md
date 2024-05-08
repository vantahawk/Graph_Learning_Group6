# Explanation and Setup for the Code of Group6 for Sheet 2

## Requirements

Use Python 3.12, other versions have not been tested and are thus not necessarily working.

Install the requirements either in a conda environment or in a virtualenv. The latter can be done like this:

---

BASH

```bash
#ON LINUX
.../group6$ python -m venv venv
.../group6$ venv/bin/activate
.../group6$ python -m pip install -r requirements.txt
```

---

BATCH/CMDLine

```batch
::ON WINDOWS cmdline, NOT powershell
...\group6> python -m venv venv
...\group6> .\venv\Scripts\activate.bat
:: in powershell use `.\venv\Scripts\Activate.ps1` instead
...\group6> python -m pip install -r requirements.txt
```

## How to run


To run code for Ex.4:

```batch
::ON WINDOWS cmdline
...\group6> python src/main.py
```

```bash
# ON LINUX
.../group6$ python src/main.py 
```

For this exercise we could reuse the color refinement but used all the iterations' distributions as discussed in the exercise sheet. This kernel is also pretty fast. 

See the notes on the [WL Test](#some-notes-to-the-wl-isomorphism-test). 

We implemented the node_labels incorrectly thus did get slightly worse results than actually possible, this is fixed now. The results with the initial labels as node labels are not much better:

dataset | w/o node labels       |   with node labels
------: | :-----------------:   | :----------------:
DD      | 0.75 ± 0.03           |   0.75 ± 0.03
Enzymes | 0.30 ± 0.07           |   0.31 ± 0.07
INC1    | 0.75 ± 0.03           |   0.77 ± 0.03


## Some more notes

The wl test first was flawed, it did not return a real multiset for the neighbors, but the accuracy loss was minor. This is now fixed. And already in the results.
