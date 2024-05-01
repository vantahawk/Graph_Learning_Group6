#Observation: all used graphs are undirected, unweighted & loopless <=> all adjacency matrices are symmetric, binary & have zero-diagonal!
#=> may use eigenvalue solution for symmetric/hermitian matrices (eigvalsh/eigsh)

#Idea: adjacency matrix A is binary => number of walks of length l from node i to j are given by (A^l)[i,j].
#=> for closed walks: i=j => diagonal entries of A^l => number of _all_ closed walks of length l given by tr(A^l).
#A symmetric => eigenvalue-decomposition: A = U D U* with D real & diagonal & U unitary => A^l = U D^l U* 
#=> tr(A^l) = tr(U D^l U*) = tr(D^l U U*) = tr(D^l) = sum of l-powers of eigenalues!

#Code: takes noticably longer for dataset DD, still reasonably quick for max_length <= ~10
#works fine for NCI1 & ENZYMES for max_length <= ~50 & possibly higher

import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh
#from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import eigsh

def algeb_closed_walk_kernel(graph: nx.Graph, max_length: int) -> np.ndarray:
    A = nx.linalg.adjacency_matrix(graph) #csr-matrix

    #sparse method:
    #Eigenvalues = eigsh(csr_matrix.asfptype(A), k=nx.number_of_nodes(Graph)-1, which='LM', maxiter=None, tol=0, return_eigenvectors=False, mode='normal')
    #dense method:
    Eigenvalues = eigvalsh(A.todense(), overwrite_a=False, check_finite=True, subset_by_index=None, driver=None) 
    #eigvalsh seems to use CPU-cores more efficiently, also slightly more accurate (based on mean deviation w.r.t. taking l-powers of A directly)

    return np.array([np.sum(np.power(Eigenvalues, l)) for l in range(2, max_length+1)])
    #returns kernel vector for walk lengths l=2 to max_length