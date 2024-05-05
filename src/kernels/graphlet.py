from networkx import Graph, adjacency_matrix
import scipy.sparse as sparse
from kernels.base import BaseKernel, KerneledGraph
import scipy
from typing import Dict, List, Tuple
import numpy as np
import os, pickle as pkl
from kernels.base import test_isomorphism
import functools
import inspect, joblib as jl, psutil
from pathlib import Path
from tqdm import tqdm

class GraphletKernel(BaseKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def compute_iso_graphs(cls, k:int):

        graphSize:Dict[int, int] = {0:1, 1:1, 2:2, 3:4, 4:11, 5:34, 6:156, 7:1044, 8:12346, 9:274668, 10:12005168}

        #for now lets fix k to 5
        if not k==5:
            raise NotImplementedError("Only graphlets of size 5 are supported for now.")
        
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
            for curr_graph in all_graphs_edges:#could be sped up by iterating over a chunk each time. There probably is a probabilitically good number of graphs to test (depending on how many were found yet)
                if test_isomorphism(iso_graphs, curr_graph, k=4).any():
                    continue
                else:
                    iso_graphs.append(curr_graph)
                    print(f"Found new isograph {len(iso_graphs)}")

                if len(iso_graphs)==graphSize[k]:
                    break


            with open(f"datasets/iso_graphs_{k}.pkl", "wb") as f:
                pkl.dump(iso_graphs, f)

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
        
        iso_graphs:List[Graph]

        with open(f"datasets/iso_graphs_{k}.pkl", "rb") as f:
            iso_graphs = pkl.load(f)
        
        #make sure all the graphs have the 5 nodes again
        for i in range(len(iso_graphs)):
            g = Graph()
            g.add_nodes_from(range(5))
            g.add_edges_from(iso_graphs[i].edges())

        #test the graphlets for isomorphism classes
        graphlets = [self.graph.subgraph(graphlets_nodes[i]) for i in range(m)]
        isomorphismClasses = test_isomorphism(iso_graphs, graphlets).T

        # print("Isomorphism classes:", isomorphismClasses.shape, "Graphlets:", len(graphlets))
        ret = np.sum(isomorphismClasses, axis=0)
        # print("Graphlet kernel vector shape:", ret.shape)
        return ret
    
    @classmethod
    def readGraphs(cls:"BaseKernel", graphPath:Path|str, **kwargs)->List[KerneledGraph]:
        """A factory method that is initialized with some data and arbitrary arguments.
        As arguments, you may provide the init arguments of the class, and the transform arguments.    
        """
        graphs:List[Graph] = []
        with open(graphPath if isinstance(graphPath, str) else graphPath.absolute(), "rb") as f:
            graphs = pkl.load(f)

        
        initKwargs = {k:v for k,v in kwargs.items() if k in inspect.signature(cls).parameters}
        transformKwargs = {k:v for k,v in kwargs.items() if k not in initKwargs}

        # print(kwargs, initKwargs, transformKwargs) # debug-print
        kernelObjects:List[BaseKernel] = [cls(graph=graph, **initKwargs) for graph in graphs]

        GraphletKernel.compute_iso_graphs(transformKwargs.get("k", 5))
        
        return jl.Parallel(n_jobs=psutil.cpu_count())(jl.delayed(kernelObjects[i].transform)(**transformKwargs) for i in tqdm(range(len(kernelObjects))))
