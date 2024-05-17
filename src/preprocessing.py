from networkx import Graph, adjacency_matrix
import numpy as np
from torch import tensor, Tensor, float32
from torch.utils.data import TensorDataset
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

    max_node_count = max([graph.number_of_nodes() for graph in graphs])

    As:List[csr_array|np.ndarray] = [adjacency_matrix(graph) for graph in graphs]
    for i,A in enumerate(As):
    #given that A is binary, just get the sqrt of the product of the degree vectors
        #compute the degrees and add one, so we have positive values only
        D:np.ndarray = np.reshape(A.copy().sum(axis=1)+1, (1,-1)) 
        D = np.sqrt(D.T @ D)
        #pad into correct size.
        R:np.ndarray = np.ndarray((max_node_count, max_node_count))
        R[:graphs[i].number_of_nodes(), :graphs[i].number_of_nodes()] = (A / D).toarray()
        
        As[i] = R

    #use float32 as the precision should be high enough and we save compute
    return Tensor(np.array(As))
    