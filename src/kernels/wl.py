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

    def transform(self, k:int=5) -> List[KerneledGraph]:
        """The weiseiler leman function.
        
        ### Args:
            k (int): The number of iterations to run the color refinement algorithm.
        
        ### Returns:
            List[KerneledGraph]: The kerneled graph vectors. A list with length #graphs of numpy arrays of shape = k x n_colors  

        ### How it works:
        Do color refinement on all graphs simultaneously.  Use the entire histogram over all iterations as the feature vector.
        """
        
        #initialize the color refinement
        colors_list:List[np.ndarray] = color_refinement(self.graphs, returnAllIterations=True, useLabels=True, n_iterations = k, return_dense = True)
        graph_vectors:List[KerneledGraph] = [np.ndarray((k, colors.shape[1])) for colors in colors_list]

        
        print("Color refinement elem shape:", colors_list[0].shape, "graph vectors elem shape:", graph_vectors[0].shape) # debug-print

        #swap the oth and 1st axis basically
        for i, colors in enumerate(colors_list):
            for g in range(colors.shape[0]):
                graph_vectors[g][i] = colors[g]


        #get the feature vector
        return graph_vectors
    
    @classmethod
    def readGraphs(cls: BaseKernel, graphPath: Path | str, **kwargs) -> List[KerneledGraph]:
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