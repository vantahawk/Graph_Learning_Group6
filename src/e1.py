from typing import List
import networkx as nx
import numpy as np
import pickle


def normalized_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    # Get the adjacency matrix as a NumPy array
    A = nx.to_numpy_array(G)

    # Compute the degrees of the nodes and adjust by adding 1
    # Sum over rows to get the degree for each node
    degrees = np.sum(A, axis=0) + 1

    # Compute the inverse square root of the degrees for use in normalization
    inv_sqrt_degrees = 1 / np.sqrt(degrees)

    # Normalize the adjacency matrix
    # (using outer product to adjust rows and columns simultaneously)
    A_tilde = np.outer(inv_sqrt_degrees, inv_sqrt_degrees) * \
        (A + np.eye(len(G)))

    return A_tilde


# load the graph from ../datasets/Citeseer_Eval/data.pkl
# with open('./datasets/NCI1/data.pkl', 'rb') as f:
#     G: List[nx.Graph] = pickle.load(f)
#     print(list(G[0].nodes(data=True))[15])
#     print(len(list(G[0].nodes(data=True))[1][1]))
#     A_tilde_optimized = normalized_adjacency_matrix(G[0])
#     print(A_tilde_optimized)


# DD node:
# (1, {'node_label': 4})

# ENZYMES node:
# (1, {'node_label': 1, 'node_attributes': [11.0, 15.887014, 37.78, -0.51, 1.701, 93.9, 4.0, 5.0, 2.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 3.0, 6.0, 2.0]})

# NCI1 node:
# (16, {'node_label': 3})

# X = np.load('Yelp/X.npy')
# print(len(X))
# print(len(X[0]))

# Y = np.load('Yelp/Y.npy')
# print(len(Y))
# print(Y)
