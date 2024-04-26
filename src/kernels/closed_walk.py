from networkx import Graph, adjacency_matrix
import scipy.sparse as sparse
from kernels.base import BaseKernel, KerneledGraph
import scipy
class ClosedWalkKernel(BaseKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, max_length:int) -> KerneledGraph:
        """The closed walk kernel function.
        
        ### Args:
            graph (Graph): The graph to kernelize.
            max_length (int): The maximum length of the closed walks to count.
        
        ### Returns:
            KerneledGraph: The kerneled graph vector. A numpy array of shape = (max_length,)

        ### How it works:
        Fix some L ∈ N. Define feature vector φ(G)∈R^L such that φ(G)_l is the number of closed walks of length l in G.
        """
        AdjacencyMatrix:sparse.csr_matrix = adjacency_matrix(self.graph)

        kernel_matrix = sparse.csr_matrix(
            (max_length, self.graph.number_of_nodes()), dtype=int)

        #start from each vertex and run a multi-walk of length max_length
        #whenever we reach our start vertex, we increment the count of the closed walk for the current length
        #
        for vertex in self.graph.number_of_nodes():
            startVector = sparse.csr_matrix((self.graph.number_of_nodes()), dtype=int)
            startVector[vertex] = 1
            for i in range(max_length):
                startVector = AdjacencyMatrix.dot(startVector)
                kernel_matrix[i, vertex] += startVector[vertex]

        #sum over the columns to get the vector of closed walks for each length<=max_length
        return kernel_matrix.sum(axis=1).toarray()