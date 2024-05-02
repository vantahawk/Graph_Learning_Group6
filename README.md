# Explanation and Setup for the Code of Group6

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
\#TODO fill this when the code is finished
To run code for Ex.4:
```batch
::ON WINDOWS cmdline
...\group6> python -m svm_main
```

Default parameters:
- for closed walk:  max_length = 20
- for graphlets:    
- for WL:           

Results for Ex.4:
Mean ± Standard Deviation of Accuracy scores (rounded in %) for 10-fold Cross-validation:

Kernel ↓ | Dataset →    DD              ENZYMES         NCI1

Closed Walk             75.72±0.03      21.0±0.32       63.77±0.1