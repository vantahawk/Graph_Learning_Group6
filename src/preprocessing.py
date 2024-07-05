from networkx import Graph, adjacency_matrix
import numpy as np
from torch import tensor, Tensor, float32
from torch.utils.data import TensorDataset
from scipy.sparse import csr_array, coo_matrix
from typing import overload, List
from scipy.linalg import eigvalsh
import networkx as nx

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
        D = D+1 #add the self-loop-> necessary for good message passing
        D = np.sqrt(D.T @ D)
        #pad into correct size.
        R:np.ndarray = np.ndarray((max_node_count, max_node_count))
        R[:graphs[i].number_of_nodes(), :graphs[i].number_of_nodes()] = (A / D).toarray()
        
        As[i] = R

    #use float32 as the precision should be high enough and we save compute
    return Tensor(np.array(As)) 
    

#basically copid from ex 1
def closed_walk_kernel(graph:nx.Graph, max_length:int)->np.ndarray:
    """The closed walk kernel function.
    
    ### Args:
        graph (nx.Graph): The graph to compute the kernel for.
        max_length (int): The maximum length of the closed walks to count.
    
    ### Returns:
        ndarray: The kerneled graph vector. A numpy array of shape = (max_length,)

    ### How it works:
    Fix some L ∈ N. Define feature vector φ(G)∈R^L such that φ(G)_l is the number of closed walks of length l in G.
    """

    eigenvalues:np.matrix = eigvalsh(adjacency_matrix(graph).todense(), overwrite_a=False, check_finite=True, subset_by_index=None, driver=None) 
    # seems to use CPU-cores more efficiently than sparse method, also slightly more accurate (based on mean deviation w.r.t. taking l-powers of A directly)
    
    eigenvalues_power = eigenvalues
    kernel_vector = np.ndarray((max_length,))
        
    for l in range(max_length):
        eigenvalues_power = np.multiply(eigenvalues_power, eigenvalues) # successively multiply eigenvalues elem.wise
        kernel_vector[l] = np.sum(eigenvalues_power)
    
    return kernel_vector # returns kernel vector for walk lengths l=2,...,max_length+1

def cwk_node_contributions(graph:nx.Graph, max_length:int=5, array_size:int|None=None, cut_duplicates:bool=True)->np.ndarray:
    """
    Compute the contribution of each node to the closed walks up to length L in the graph.
    
    Args:
    - graph (nx.Graph): input graph
    - length (int): int, maximum length of walks to consider
    
    Returns:
    - node_scores: numpy array, scores for each node based on their contribution to closed walks.
    """
    # Compute eigenvalues and eigenvectors of the adjacency matrix
    A = adjacency_matrix(graph).todense()
    eigvals, eigvecs = np.linalg.eig(A)
    node_count = graph.number_of_nodes()
    # Initialize an array to store the scores for each node
    node_scores = np.zeros((node_count if array_size == None else array_size, max_length))
    
    # Compute contributions for each node
    for k in range(max_length):
        # do this in parallel using numpy
        node_scores[:node_count, k] = np.sum((eigvecs ** 2) * (eigvals ** k), axis=1)

    #round to integer (make low nonzero values e-14 disappear)
    node_scores = np.floor(node_scores).astype(int)

    if cut_duplicates:
        #remove duplicates, meaning e.g. for a length 12, remove all the subfactor influences
        for i in range(2, max_length//2+1):
            for j in range(2*i, max_length+1, i):
                node_scores[:, j] -= node_scores[:, i]

        #quite interesting in how the duplicates are removed:
        # 1. for each possible length that has an influence on another length
        # 2. for each node that it has an influence on
        # 3. remove the influence of the first length on the second length
        # that way, we remove only the influences that really are there, if done in other order, we would remove influences twice

    return node_scores