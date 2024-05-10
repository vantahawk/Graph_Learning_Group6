from networkx import Graph, adjacency_matrix
import numpy as np
from torch import tensor, Tensor, float32
from scipy.sparse import csr_array, coo_matrix
from typing import overload, List

@overload
def normalized_adjacency_matrix(graph: Graph)->Tensor:
    """
    Computes the normalized adjacency matrix of a graph.
    
    Args:
        graph (Graph): A NetworkX graph object.
    
    Returns:
        (torch.sparse.FloatTensor): A normalized adjacency matrix.
    """

    pass

@overload
def normalized_adjacency_matrix(graphs: List[Graph])->Tensor:
    """
    Computes the normalized adjacency matrix of a list of graphs.
    
    Args:
        graphs (List[Graph]): A list of NetworkX graph objects.
    
    Returns:
        (torch.sparse.FloatTensor): A normalized adjacency matrix.
    """

    pass

def normalized_adjacency_matrix(graphs: Graph|List[Graph])->Tensor:
    if not isinstance(graphs, list):
        graphs = [graphs]

    As:List[csr_array|coo_matrix] = [adjacency_matrix(graph) for graph in graphs]
    
    for i,A in enumerate(As):
    
    #given that A is binary, just get the sqrt of the product of the degree vectors
        D:np.ndarray = np.reshape(A.copy().sum(axis=1), (1,-1))
        D = np.sqrt(D.T @ D)
        As[i] = A/D

    return tensor(np.array([A.toarray() for A in As]), dtype=float32).to_sparse()
    