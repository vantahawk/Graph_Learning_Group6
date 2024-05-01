# Observation:
# all used graphs are undirected, unweighted & loopless <=> all adjacency matrices are symmetric, binary & have zero-diagonal!
# => may use eigenvalue solution for symmetric/hermitian matrices (eigvalsh/eigsh)

# Idea:
# adjacency matrix A is binary => number of walks of length l from node i to j are given by (A^l)[i,j]
# => for closed walks: i=j => diagonal entries of A^l => number of _all_ closed walks of length l given by tr(A^l)
# A symmetric => eigenvalue-decomposition: A = U D U* with D real & diagonal & U unitary => A^l = U D^l U* 
# => tr(A^l) = tr(U D^l U*) = tr(D^l U U*) = tr(D^l) = sum of l-powers of eigenalues!
# computing A^l up to L directly: O((L-1) n^3) elem. op.s
# computing l-powers of eigenvalues up to L: O((L-1) n) elem. op.s
# => potential speed up by factor n^2 !

# Computation (using eigvalsh):
# takes noticably longer for dataset DD, still reasonable (~40s) for max_length <= ~20
# only second(s) for NCI1 & ENZYMES for max_length <= ~100 & possibly higher
# for NCI1: overflow at max_length >= ~600
# for ENZYMES: overflow at max_length >= ~400

import numpy as np
import networkx as nx
from networkx.linalg import adjacency_matrix
from scipy.linalg import eigvalsh
#from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import eigsh

def algeb_closed_walk_kernel(graph: nx.Graph, max_length: int) -> np.ndarray:
    # eigenvalues of adjacency matrix; choose:
    # sparse method:
    #eigenvalues = eigsh(csr_matrix.asfptype(adjacency_matrix(graph)), k=nx.number_of_nodes(Graph)-1, which='LM', maxiter=None, tol=0, return_eigenvectors=False, mode='normal')
    # or dense method:
    eigenvalues = eigvalsh(adjacency_matrix(graph).todense(), overwrite_a=False, check_finite=True, subset_by_index=None, driver=None) 
    # seems to use CPU-cores more efficiently than sparse method, also slightly more accurate (based on mean deviation w.r.t. taking l-powers of A directly)
    
    eigenvalues_power = eigenvalues
    kernel_vector = []
        
    for l in range(2, max_length+1):
        eigenvalues_power = np.multiply(eigenvalues_power, eigenvalues) # successively multiply eigenvalues elem.wise
        kernel_vector.append(np.sum(eigenvalues_power))
    
    return np.array(kernel_vector) # returns kernel vector for walk lengths l=2 to max_length (L)