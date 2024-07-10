# Link Prediction

This solution contains the implementation of a solution to predict missing edge labels dataset using the TransE model. The solution is based on the paper [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) by Bordes et al.

## Dataset

The expected input dataset has the exemplary form:

```python
# edges: (8284, 208, {'edge_label': 29, 'id': 28021})
# nodes: (56, {})
```

where $edge\_label \in \{\text{None}, 0, 1, ..., 29\}$.

## Requirements

- Python 3.x
- numpy
- networkx
- pickle

## Solution

The solution involves Preprocessing the graph data (loading with pickle, splitting into training and test sets, and converting nodes and relationships to indices), implementing the TransE model, training the model on the training data, predicting missing labels for the test edges, and saving the predictions to a file.

To run the solution:

1. Make sure your dataset is in `$PROJECT_ROOT/datasets/LINK/data.pkl`.

2. Run the script:

   ```bash
   pip install -r requirements.txt
   python3 link_prediction.py
   ```

3. Check the output file `LINK-Predictions.pkl` for the predicted labels that are given in the order they appear in the training set.
