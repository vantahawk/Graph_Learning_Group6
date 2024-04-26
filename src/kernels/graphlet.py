from networkx import Graph, adjacency_matrix
import scipy.sparse as sparse
from kernels.base import BaseKernel, KerneledGraph
import scipy
from typing import Dict
import numpy as np

class GraphletKernel(BaseKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, k:int=5, m:int|None=None) -> KerneledGraph:
        """The graphlet kernel function.
        
        ### Args:
            graph (Graph): The graph to kernelize.
            k (int): The size of the graphlets to sample.
            m (int): The number of graphlets to sample. If None, sample #V(G)/k graphlets.
                    #TODO make sure that the default is something sensible.
        ### Returns:
            KerneledGraph: The kerneled graph vector.

        ### How it works:
        Uniformly sample m random subgraphs (graphlets) of size k from G.

        We will use k = 5. There are 34 distinct graphs with 5 nodes.

        Define the feature vector φ(G) ∈R^34 such that φ(G)_i is the number of sampled
        graphlets of isomorphism class i.
        """
        if k not in range(11):
            raise ValueError("The graphlet kernel only supports graphlets of size 0..10")

        graphSize:Dict[int, int] = {0:1, 1:1, 2:2, 3:4, 4:11, 5:34, 6:156, 7:1044, 8:12346, 9:274668, 10:12005168}

        #for now lets fix k to 5
        if not k==5:
            raise NotImplementedError("Only graphlets of size 5 are supported for now.")
    
        m = m if m else self.graph.number_of_nodes()//k

        #sample m random graphlets of size k
        graphlets:np.ndarray = np.ndarray((m, k))
        graphlets = np.random.choice(self.graph.number_of_nodes(), (m, k))

        #test for isomorphism classes
        isomorphismClasses:np.ndarray = np.zeros((m, 34))
        



        
