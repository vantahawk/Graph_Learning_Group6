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
...\group6> python src/main.py --kernel <KERNEL>
```

```bash
# ON LINUX
.../group6$ python src/main.py --kernel <KERNEL>
```

For more info check the `--help` page, like so: `python src/main.py --help`

Default parameters:

- for closed walk: max_length = 20
- for graphlets: as discussed in the exercise
- for WL: as in the exercise sheet

Results for Ex.4:

Mean ± Standard Deviation of Accuracy scores (rounded in %) for 10-fold Cross-validation:

| Kernel ↓ , Dataset → | DD         | ENZYMES    | NCI1       |
| :------------------- | :--------- | :--------- | :--------- |
| Closed Walk          | 76.23±3.26 | 23.00±5.25 | 64.89±2.84 |
| Graphlet             | 71.99±0.16 | 16.83±0.50 | 59.85±0.11 |
| WL                   | 76.23±2.81 | 24.33±4.95 | 68.78±3.21 |


Best Params | DD        | Enzymes   | NCI1
----------- | --------- | --------- | ----------
Closed Walk |           | C: 5, break_ties: True, cache_size: 200, coef0: 1.0, decision_function_shape: 'ovr', degree: 5, gamma: 'scale', kernel: 'poly', max_iter: -1, tol: 0.0001 <br>-> 25.0±5.16 |     
Graphlet    |           |           |
WL          |           |           |
---

## Ex.1:

### Observation:

- all used graphs are undirected, unweighted & loopless <=> all adjacency matrices are symmetric, binary & have zero-diagonal!

- => may use eigenvalue solution for symmetric/hermitian matrices (eigvalsh/eigsh)

### Idea:

- adjacency matrix A is binary => number of walks of length l from node i to j are given by (A^l)[i,j]

- => for closed walks: i=j => diagonal entries of A^l => number of _all_ closed walks of length l given by tr(A^l)

- A symmetric => eigenvalue-decomposition: A = U D U' with D real & diagonal & U unitary => A^l = U D^l U'

- => tr(A^l) = tr(U D^l U') = tr(D^l U' U) = tr(D^l) = sum of l-powers of eigenvalues!

- computing A^l up to L directly: O((L-1)\*n^3) elem. op.s

- computing l-powers of eigenvalues up to L: O((L-1)\*n) elem. op.s

- => potential speed up by factor n^2 (excluding eigenproblem solving, which itself likely scales with some power of n)!

### Computation (using eigvalsh):

- takes noticably longer for dataset DD, still manageable (~45s) for max_length <= ~20, thus we chose max_length = 20 as default

- only second(s) for ENZYMES & NCI1 for max_length <= ~100 & possibly higher

- for ENZYMES: overflow at max_length >= ~400

- for NCI1: overflow at max_length >= ~600

### Comparison:

- closed walk kernel does not reach mean accuracies as high as in the exercise sheet or the paper, comes close though for dataset DD

## Some notes to the WL Isomorphism test:

For implementing this as fast as possible, we decided on a fast hash function: xxhash for 32bit hashes. We thus do not need to save hashes in some data structure, the hash function with a given seed is our "datastructure". As seed we use some small hash of an abstraction of the initial graph coloring array (all graphs are in there).

For computing the multiset, we use a histogram representation, that first densifies the coloring before computing the histogram to save memory (even allow the code to run). The histogram in this case only contains the values of the neigbors, but among all possible colors.

We also use this densified histogram representation for our colorings representation on which the isomorphism test is computed. (This is a complete histogram among all graphs and nodes.)

## Ex.2:

We precompute the isomorphism classes for the given k, then use these as references to test our graphlets against. As a isomorphism test, we use the wl-test, which is necessary for exercise 3 with $\#iterations = k-1$.

The precomputation is pretty fast for k=5, the kernel then takes some mediocre time and the fitting with hpo takes the most time.

The results we get do not seem to match the results from the paper. The algorithm however seems to be correct, at least it seems like that.

## Ex.3:

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
