This directory contains the graph datasets for the first exercise.

The subdirectory "datasets" contains the three main datasets you will focus on for now: DD [1], NCI1 [2] and ENZYMES [3].
The datasets are provided as pickled NetworkX graph objects.
For example, the following code will load the NCI1 dataset:

```
import pickle
with open('datasets/NCI1/data.pkl', 'rb') as f:
    graphs = pickle.load(f)
```

The graphs are returned as a list of NetworkX graph objects (https://networkx.github.io/).

The node and graph labels are stored as attributes in the graph objects. 
For example, you may use the following commands to extract this information for one graph in a dataset:

```
G = graphs[0]
graph_label = G.graph['label']
node_labels = [node[1]['node_label'] for node in G.nodes(data=True)]
```

In this sheet, the graph label is the prediction target while the node labels are part of the input (if the kernel supports node labels).
Note that the graphs of the Enzymes dataset do not just have discrete node labels but also additional real-numbered vectors as node attributes. 
These are not relevant for the first exercise.

References:

[1] P. D. Dobson and A. J. Doig. "Distinguishing enzyme structures from non-enzymes without alignments."

[2] K. M. Borgwardt, C. S. Ong, S. Schoenauer, S. V. N. Vishwanathan, A. J. Smola, and H. P. Kriegel. "Protein function prediction via graph kernels"

[3] N. Wale and G. Karypis. "Comparison of descriptor spaces for chemical compound retrieval and classification"
