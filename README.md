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

- for closed walk: max_length = 20
- for graphlets:
- for WL:

Results for Ex.4:

Mean ± Standard Deviation of Accuracy scores (rounded in %) for 10-fold Cross-validation:

| Kernel ↓ , Dataset → | DD         | ENZYMES   | NCI1       |
| :------------------- | :--------- | :-------- | :--------- |
| Closed Walk          | 75.72±1.72 | 21.0±5.64 | 63.77±3.14 |
| Graphlet             | 73.27±3.79 |           | 60.61±3.43 |
| WL                   |            |           |            |

---

Ex.1:

Observation:

- all used graphs are undirected, unweighted & loopless <=> all adjacency matrices are symmetric, binary & have zero-diagonal!

- => may use eigenvalue solution for symmetric/hermitian matrices (eigvalsh/eigsh)

Idea:

- adjacency matrix A is binary => number of walks of length l from node i to j are given by (A^l)[i,j]

- => for closed walks: i=j => diagonal entries of A^l => number of _all_ closed walks of length l given by tr(A^l)

- A symmetric => eigenvalue-decomposition: A = U D U' with D real & diagonal & U unitary => A^l = U D^l U'

- => tr(A^l) = tr(U D^l U') = tr(D^l U' U) = tr(D^l) = sum of l-powers of eigenvalues!

- computing A^l up to L directly: O((L-1)\*n^3) elem. op.s

- computing l-powers of eigenvalues up to L: O((L-1)\*n) elem. op.s

- => potential speed up by factor n^2 (excluding eigenproblem solving, which itself likely scales with some power of n)!

Computation (using eigvalsh):

- takes noticably longer for dataset DD, still manageable (~45s) for max_length <= ~20, thus we chose max_length = 20 as default

- only second(s) for ENZYMES & NCI1 for max_length <= ~100 & possibly higher

- for ENZYMES: overflow at max_length >= ~400

- for NCI1: overflow at max_length >= ~600

Comparison:

- closed walk kernel does not reach mean accuracies as high as in the exercise sheet or the paper, comes close though for dataset DD

Ex.2:

This kernel unfortunately did not scale up well for the datasets, so results took too long. Moreover, scikit threw a numpy error for the ENZYMES datasets regarding the data type in the np.array method, which we could not get our heads around.

Idea:

Our kernel works by checking for each random graphlet if it is isomorphic to one of the graphlets from the on-the-fly created list of graphlets. If it is, the corresponding entry in the kernel matrix is increased by 1.

Observation:

This kernel is very slow, as it has to check for each graphlet if it is isomorphic to one of the graphlets from the list. This is a very expensive operation, as it has to check for each permutation of the nodes of the graphlet if it is isomorphic to the graphlet from the list. We thought of an optimization which promotes the graphlet that has a high number of occurences to the start of the list, so that the kernel has to check for this graphlet first. This way, the kernel can stop checking for isomorphisms as soon as it finds a match. But we did not implement this optimization, as we did not have enough time.

Ex.3:

This kernel is also too slow to be computed in a reasonable amount of time. We did not get any results for this kernel. It also threw an error in the svm_main.py file, which we could not resolve.

Idea:

We basically implemented a simplified intuitive version for the kernel that leverages an iteration function `refine_colors` which is also too slow. We did not have enough time to implement the algebraic approach that was mentioned in the exercise sheet.
