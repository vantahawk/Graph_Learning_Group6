from networkx import Graph, adjacency_matrix
import scipy.sparse as sparse
from kernels.base import BaseKernel, KerneledGraph
import scipy
from typing import Dict, List, Tuple
import numpy as np
import os, pickle as pkl
from kernels.base import test_isomorphism
import functools

class GraphletKernel(BaseKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, k:int=5, m:int|None=None) -> KerneledGraph:
        """The graphlet kernel function.
        
        ### Args:
            k (int): The size of the graphlets to sample.
            m (int): The number of graphlets to sample. If None, sample #V(G)/(k/2) graphlets.
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
    
        m = m if m else self.graph.number_of_nodes()//(k//2)

        #sample m random graphlets of size k
        graphlets_nodes:np.ndarray = np.ndarray((m, k))
        graphlets_nodes = np.random.choice(self.graph.number_of_nodes(), (m, k))

        #test for isomorphism classes
        isomorphismClasses:np.ndarray[bool, bool] = np.zeros((m, graphSize[k]))
        
        #generate all possible isomorphism graphs given k
        #we don't use labels, so only edges matter=>use edges
        iso_graphs:List[Graph] = []
        if not os.path.exists(f"datasets/iso_graphs_{k}.pkl"):
            #generate all possible graphs, then sort out duplicates
            all_graphs_edges:List[Graph] = []

            @functools.cache
            def edgeindex(k:int, i:int, j:int)->int:
                return (k-i)*i + i*(i-1)//2 + j-i-1

            base_graph:Graph = Graph()
            base_graph.add_nodes_from(range(k))
            #generate all possible graphs with duplicates
            for g in range(2**(k*(k-1)//2)):
                bin_g = bin(g)[2:].zfill(k*(k-1)//2)
                edge_list:List[Tuple[int, int]] = []
                for i in range(k):
                    for j in range(i+1, k):
                        if int(bin_g[edgeindex(k,i,j)]):
                            edge_list.append((i,j))
                curr_graph:Graph = base_graph.copy()
                curr_graph.add_edges_from(edge_list)
                all_graphs_edges.append(curr_graph)

            #remove duplicates dynamically
            iso_graphs.append(all_graphs_edges.pop(0))
            for edge_list in all_graphs_edges:#could be sped up by iterating over a chunk each time. There probably is a probabilitically good number of graphs to test (depending on how many were found yet)
                curr_graph = Graph(edge_list)
                if test_isomorphism(iso_graphs, curr_graph).any():
                    continue
                else:
                    iso_graphs.append(curr_graph)
                    print(f"Found new isograph {len(iso_graphs)}")

                if len(iso_graphs)==graphSize[k]:
                    break


            with open(f"datasets/iso_graphs.pkl", "wb") as f:
                pkl.dump(iso_graphs, f)
        else:
            with open(f"datasets/iso_graphs.pkl", "rb") as f:
                iso_graphs = pkl.load(f)

        #test the graphlets for isomorphism classes
        graphlets = [self.graph.subgraph(graphlets_nodes[i]) for i in range(m)]
        isomorphismClasses = test_isomorphism(iso_graphs, graphlets).T

        return isomorphismClasses
