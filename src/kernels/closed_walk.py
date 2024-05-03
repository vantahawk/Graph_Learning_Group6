from networkx import Graph, adjacency_matrix
import scipy.sparse as sparse
from kernels.base import BaseKernel, KerneledGraph
import scipy
import numpy as np
from scipy.linalg import eigvalsh
class ClosedWalkKernel(BaseKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, max_length:int) -> KerneledGraph:
        """The closed walk kernel function.
        
        ### Args:
            max_length (int): The maximum length of the closed walks to count.
        
        ### Returns:
            KerneledGraph: The kerneled graph vector. A numpy array of shape = (max_length,)

        ### How it works:
        Fix some L ∈ N. Define feature vector φ(G)∈R^L such that φ(G)_l is the number of closed walks of length l in G.
        """
        # AdjacencyMatrix:sparse.lil_matrix = sparse.lil_matrix(adjacency_matrix(self.graph))

        # kernel_matrix = sparse.lil_matrix(
        #     (max_length, self.graph.number_of_nodes()), dtype=int)

        # #start from each vertex and run a multi-walk of length max_length
        # #whenever we reach our start vertex, we increment the count of the closed walk for the current length
        # #
        # # print(f"AdjacencyMatrix: {AdjacencyMatrix.shape}, startVector: {sparse.csr_matrix((self.graph.number_of_nodes(),1), dtype=int).shape}")

        # for vertex in range(self.graph.number_of_nodes()):
        #     startVector = sparse.lil_matrix((self.graph.number_of_nodes(),1), dtype=int)
        #     startVector[vertex] = 1
        #     for i in range(max_length):
        #         startVector = AdjacencyMatrix.dot(startVector)
        #         kernel_matrix[[i], [vertex]] += startVector[[vertex]]

        # #sum over the columns to get the vector of closed walks for each length<=max_length
        # ret:np.matrix =  kernel_matrix.sum(axis=1) #should be a numpy array already
        # # print(f"ret: {ret.shape}, ret-type: {type(ret)}")
        # return np.array(ret).ravel() #make into np.ndarray

        eigenvalues:np.matrix = eigvalsh(adjacency_matrix(self.graph).todense(), overwrite_a=False, check_finite=True, subset_by_index=None, driver=None) 
        # seems to use CPU-cores more efficiently than sparse method, also slightly more accurate (based on mean deviation w.r.t. taking l-powers of A directly)
        
        eigenvalues_power = eigenvalues
        kernel_vector = np.ndarray((max_length,))
            
        for l in range(max_length):
            eigenvalues_power = np.multiply(eigenvalues_power, eigenvalues) # successively multiply eigenvalues elem.wise
            kernel_vector[l] = np.sum(eigenvalues_power)
        
        return kernel_vector # returns kernel vector for walk lengths l=2,...,max_length+1
