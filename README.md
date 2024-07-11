# Explanation and Setup for the Code of Group 6 for Sheet 5

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
---

## HOLU

The entry point for HOLU is `...\group6> python main.py`. The other files required for it lie in `src`, `src/geomConvert` and `helpful_extra_features`.

---

## CITE

### How to run code for CITE

```batch
::ON WINDOWS cmdline
...\group6> python src/cite/cite.py
```

or

```bash
# ON LINUX
.../group6$ python src/cite/cite.py
```

...runs the model in full *prediction mode* (see Idea & Method - Schedule) with the default parameters. To run the *cross-validation mode* (see Idea & Method - Schedule) instead, set the flag `-cv`. Moreover with `-k <integer>` you can also specify the number of splits/folds in the cross-validation (default: 12), and with `-epochs <integer>` the number of epochs (per split). Moreover `-s` allows you to skip the embedding computation if the relevant parameters have not been changed since the last run (see Idea & Method - Note).
All other parameters have to be set/changed manually in `cite.py` below `if __name__ == "__main__":`.


### Idea & Method

Our model for node classification on the single graph in CITE combines the graph learning approaches from all the previous sheets with some added twists. The core model is a GNN which is segmented into two input stages: Into the primary input stage (a GNN layer stack) the node_attributes of (subgraphs of) the graph are fed. For the second input stage (a multi-layer perceptron) the node-level outputs from the primary stage are concatenated with two external embeddings - a special node2vec-embedding and a node-level closed-walk-kernel embedding - along the node-axis before being fed through. The final output are the one-hot-encoded node_labels predicted on the given (sub)graph. The model trains with Adam optimizer over several epochs using cross-entropy as loss. Below we explain the novel details of each component of the model and their motivation, derivation and implementation.

- __GNN:__ Implemented in `src/cite/gnn.py` & `src/cite/layer.py`<br>
Each GNN layer of the primary input stage uses scatter operations for message passing like in sheet 3 but since there are no edge labels the message-layer (M) is missing as a designated module. Instead a degree-normalization is applied during message-passing which mirrors the behaviour of the normalized adjacency matrix as in sheet 2. Thus if one uses `scatter_sum` for message passing (as is the default), one effectively yields a GCN.<br>
There is an additional parameter `n_pass` which is the number of message-passing steps performed within each GNN layer. E.g. for `n_pass = 3` each node would receive messages from 3 edges removed, which in the GCN context implies multiplying the normalized adjacency matrix 3 times wihtin each layer. The default is `n_pass = 1` which turned out to likely be sufficient afterall.

- __N2V:__ Implemented in `src/cite/random_trees.py` & `src/cite/node2vec.py`<br>
Instead of random pq-walks like in sheet 4, this version of node2vec trains on batches of random *trees*, here called *p-trees*. Building each such tree starts by uniformly sampling a start node from the whole given (sub)graph, and then uniformly sampling & adding new, unique nodes from each subsequent *frontier* (i.e. the subset of potential next child nodes) until $m \in \mathbb{N}$ nodes have been collected ($m+1$ in total). The parameter $p \in (0,1]$ is the (rounded up) relative share of new nodes to be drawn from each frontier. Thus $p$ controls the general structure of the emerging tree in a somewhat analogous manner to $p$ & $q$ in the case of pq-walks: Small $p$ tends to yield a sparse but broad tree (depth first) whereas large $p$ tends to yield a dense but localized tree (breadth first).<br>
Furthermore $m_{ns} \in \mathbb{N}$ unique *negative samples* are uniformly sampled from the complement of the tree nodes. Analogously to sheet 4, the start node, tree nodes & negative samples are then concatenated into one tree array and several tree arrays are stacked into a tree batch. Note that "trees" are here only represented in terms of their nodes but not edges. Thus nothing is implied about the presence or absence of cycles, but that is irrelevant for our *node-level* purposes.<br>
The node2vec training schedule is naively adapted from that described in sheet 4, i.e. using the same log-probability loss function, but implemented in a simplified form.<br>
In the end the N2V-embedding is normalized via standard score over all its entries.

- __CW:__ Implemented in `src/cite/closed_walks.py`<br>
Analogously to sheet 1, the *node*-level CW-embedding counts the number of closed walks of lengths $j \in \{2,...,l\}$ from and to each node (embedding dimension $l-1$). Given our binary-valued adjacency this amounts to the diagonal elements of its $j$-th powers. Given the previously expected inefficiency of computing the neccessary matrix multiplications directly (as it also computes all non-diagonal elements), we initially went for a *spectral method* which instead solves the eigenproblem of the adjacency matrix and then uses its eigenvalues & -vectors to compute said diagonal elements alone.<br>
However the `scipy.sparse` implementation for solving the eigenproblem turned out to be too slow even when only computing a few hundred eigenpairs, and became dramatically slower in the thousands (yet there are 13k+ nodes in the full graph). So even if the implementation were stable, it would still only yield an approximation based on a fraction of eigenvalues with the largest magnitude.<br>
However on top of that, computing the powers of eigenvalues also seemed to be very numerically instable even in `float64`, with values exploding to infinity for $l > 8$. Thus instead, we went with direct-multiplication-method afterall as it is exact, numerically stable and actually still faster. For reference the spectral method is still left decommented in `closed_walks.py`<br>
In the end the CW-embedding is normalized with a logarithm so as to compress its scale variation, since entries scale roughly exponentially with $k \in {2,...,l}$.

- __Schedule:__ The overall model can be run in 2 modes as implemented in `cite.py`:<br>
__(1)__ A *cross-validation mode*, which splits the *known* subgraph (with node_labels given, starting at node 1000) into `k` (default: 12) validation subgraphs with each one's complement being the training subgraph. The nodes of each validation subgraph are uniformly sampled from the known graph. After running `n_epochs_cv` epochs on each split/fold the accuracy w.r.t. to the predicted vs. the true node_labels in the validation subgraph are printed. In the end the mean and standard deviation of validation accuracies over all splits is printed.<br>
__(2)__ In *prediction mode* (set as default) the model is trained over `n_epochs_pred` epochs on the whole *known* subgraph and then used to predict the labels of the *unknown* subgraph (nodes 0-999). The predicted labels are then saved in `CITE-Predictions.pkl`.<br>
__Note:__ While the GNN itself runs fairly fast on CUDA at least, computing the external embeddings may take a while. Especially N2V may take over half an hour in single-process (we could not get a multiprocessing or CUDA-solution to work until now). Thus it may be more convenient to skip those computations if their required parameters have remained the same since the previous run(s).
To this end you may want to set the flag `-s` in the command line in case you want to do several runs with the same parameters.
(This will save/load local copies of the embeddings and parameters last modified and proceed with those instead.) Though for the sake of comprehensiveness both embeddings will be computed anew by default.

- __Sparse Graph:__ `sparse_graph.py` contains a custom class that returns a sparse representation for a given (sub)graph, including everything required during all procedures: number of nodes, adjacency matrix, the vector for degree normalization (`degree_factors`), the node indices (`node_idx`), the edges indexed by their nodes (`edge_idx`), node_attributes and _labels (if available), etc.


### Chosen Hyperparameters

These current default parameters were chosen by trial-and-error and intuitions from previous exercises.

- schedule:<br>
    - k = 12
    - n_epochs_cv = 25
    - n_epochs_pred = 25
- GNN:<br>
    - n_MLP_layers = 5
    - dim_MLP = 200
    - n_GNN_layers = 5
    - dim_between = 200 (dim. between GNN layers)
    - dim_U = 200
    - n_U_layers = 2
    - n_pass = 1
    - scatter_type = 'sum'  (for message-passing)
    - lr_gnn = 0.001 (GNN learning rate)
- N2V/p-trees:<br>
    - p = 0.5
    - m = 10
    - m_ns = 19
    - dim_n2v = 180  (N2V embedding dim.)
    - batch_size = 100
    - n_batches = 10
    - lr_n2v = 0.01  (N2V learning rate)
- CW:<br>
    - l = 21


## Cross-Validation Results

Mean ± StD of Accuracy (rounded in %) achieved with parameters above:   68.62 ± 1.77


## Discussion

Overall the results turned out to be surprisingly decent, although surely not highly impressive. However even better results may well be achieved if an HPO were run on some of the parameters like: higher numbers of layers, dimensions, tree batch size and number thereof, as well as better suited p, m, m_ns & l.

Somewhat improbably though (at lest last time we checked) the predicted node_labels mostly contained '0' and '1', very little '3' and no '2'. We would reckon this was due to an error on our part, although that remains to be seen we suppose.

Given the parameters above, we observed that validation accuracies usually peaked at around 25 epochs and slightly dropped afterwards; likely due to overfitting since training accuracies kept on growing steadily

For some unknown reason, sampling the p-trees was much slower than for pq-walks, even on single-process CPU.

Overall the scattershot approach of combining all mentioned approaches and concatenating their inputs at least seems promising in principle.

---

### LINK

This solution contains the implementation of a solution to predict missing edge labels dataset using the TransE model. The solution is based on the paper [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) by Bordes et al.

### Dataset

The expected input dataset has the exemplary form:

```python
# edges: (8284, 208, {'edge_label': 29, 'id': 28021})
# nodes: (56, {})
```

where $edge\_label \in \{\text{None}, 0, 1, ..., 29\}$.

### Requirements

- Python 3.x
- numpy
- networkx
- pickle

### Solution

The solution involves Preprocessing the graph data (loading with pickle, splitting into training and test sets, and converting nodes and relationships to indices), implementing the TransE model, training the model on the training data, predicting missing labels for the test edges, and saving the predictions to a file.

To run the solution:

1. Make sure your dataset is in `$PROJECT_ROOT/datasets/LINK/data.pkl`.

2. Run the script:

   ```bash
   pip install -r requirements.txt
   python3 link_prediction.py
   ```

3. Check the output file `LINK-Predictions.pkl` for the predicted labels that are given in the order they appear in the training set.

---

### Note on Exercise Split

Benedict worked on HOLU, David on CITE and Ahmet on LINK. Unfortunately Benedict could not contribute `HOLU-Predictions.pkl` in time for the Moodle upload, since he ran into cluster space limits and a subsequent slowdown, as well as computation errors at the last minute. He will hopefully have forwarded his file by mail shortly after.
