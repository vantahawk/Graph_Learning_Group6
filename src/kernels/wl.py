from pathlib import Path
from networkx import Graph, adjacency_matrix
import scipy.sparse as sparse
from kernels.base import BaseKernel, KerneledGraph, color_refinement
import scipy
from typing import Dict, List, Tuple
import numpy as np
import os, pickle as pkl, inspect


class WLKernel(BaseKernel):
    def __init__(self, graphs:List[Graph], **kwargs):
        self.graphs = graphs

    def transform(self, k:int=5) -> sparse.csr_matrix:
        """The weiseiler leman function.
        
        ### Args:
            k (int): The number of iterations to run the color refinement algorithm.
        
        ### Returns:
            List[KerneledGraph]: The kerneled graph vectors. A list with length #graphs of numpy arrays of shape = k x n_colors  

        ### How it works:
        Do color refinement on all graphs simultaneously.  Use the entire histogram over all iterations as the feature vector.
        """
        # print("max graph size:", max([g.number_of_nodes() for g in self.graphs]))
        #initialize the color refinement
        colors_list:List[np.ndarray] = color_refinement(self.graphs, returnAllIterations=True, useLabels=True, n_iterations = k, return_dense = True, n_jobs=-1)
        # print("Color refinement shapes:", *[colors_list[i].shape for i in range(len(colors_list))]) # debug-print
        # print("how many zeroes:", len(np.where(colors_list[2]==0)))

        # print("Color refinement elem shape:", colors_list[0].shape, "graph vectors elem shape:", graph_vectors[0].shape) # debug-print

        #get the feature vector
        return sparse.lil_matrix(np.concatenate(tuple(colors_list), axis=1)).tocsr()
    
    @classmethod
    def readGraphs(cls: BaseKernel, graphPath: Path | str, **kwargs) -> sparse.csr_matrix:
        """A factory method that is initialized with some data and arbitrary arguments.
        As arguments, you may provide the init arguments of the class, and the transform arguments.    
        """
        graphs:List[Graph] = []
        with open(graphPath if isinstance(graphPath, str) else graphPath.absolute(), "rb") as f:
            graphs = pkl.load(f)

        
        initKwargs = {k:v for k,v in kwargs.items() if k in inspect.signature(cls).parameters}
        transformKwargs = {k:v for k,v in kwargs.items() if k not in initKwargs}

        kernelObject:WLKernel = cls(graphs, **initKwargs)
        
        return kernelObject.transform(**transformKwargs)