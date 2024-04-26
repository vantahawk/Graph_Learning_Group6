import abc
import networkx as nx
import pickle as pkl
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import NewType, List, overload, Tuple

KerneledGraph = NewType("KerneledGraph", np.ndarray)
"""A type hint for the kerneled graph, which is a numpy array."""

class BaseKernel():

    @abc.abstractmethod
    def __init__(self, graph:nx.Graph, **kwargs):
        self.graph = graph
        ...

    @abc.abstractmethod
    def transform(self, graph: nx.Graph, **kwargs)->KerneledGraph:
        """the important function that implements the kerneling method and returns the kerneled graph."""
        raise NotImplementedError("The method transform must be implemented in a subclass.")

    @classmethod
    def readGraph(cls:"BaseKernel", graphPath:Path|str, **kwargs)->KerneledGraph:
        """A factory method that is initialized with some data and arbitrary arguments.
        As arguments, you may provide the init arguments of the class, and the transform arguments.    
        """
        graph: nx.Graph = None
        with open(graphPath if isinstance(graphPath, str) else graphPath.absolute(), "rb") as f:
            graph = pkl.load(f)

        
        initKwargs = {k:v for k,v in kwargs.items() if k in inspect.signature(cls).parameters}
        transformKwargs = {k:v for k,v in kwargs.items() if k not in initKwargs}

        kernelObject:BaseKernel = cls(graph, initKwargs)
        
        return kernelObject.transform(graph, **transformKwargs)
    
    
    def show(self, saveOnly:bool=False, saveToPath:str|None=None):
        nx.draw_networkx(self.graph)

        if saveToPath:
            plt.savefig(saveToPath)
            if saveOnly:
                plt.close()
                return
        
        plt.show()

@overload
def test_isomorphism(ref_graph:nx.Graph, test_graph:nx.Graph)->bool:
    """Test the isomorphism of the test graph against the reference graph.
    
    ### Args:
        ref_graph (nx.Graph): The reference graph.
        test_graph (nx.Graph): The test graph.
    ### Returns:
        bool: True if the graphs are isomorphic, False otherwise.
    """
    pass

@overload
def test_isomorphism(ref_graphs:List[nx.Graph], test_graph:nx.Graph)->np.ndarray[bool]:
    """Test the isomorphism of the test graphs against the reference graphs.
    
    ### Args:
        ref_graphs (List[nx.Graph]): The reference graphs.
        test_graph (nx.Graph): The graph to test to what reference graphs it is isomorphic.
    
    ### Returns:
        np.ndarray[bool]: A boolean array of length len(ref_graphs) where each entry is True if the test graph is isomorphic to the corresponding reference graph.
    """
    pass

@overload
def test_isomorphism(ref_graphs:List[nx.Graph], test_graphs:List[nx.Graph])->np.ndarray[Tuple[bool, bool]]:
    """Test the isomorphism of the test graphs against the reference graphs.
    
    ### Args:
        ref_graphs (List[nx.Graph]): The reference graphs.
        test_graphs (List[nx.Graph]): The graphs to test for isomorphism.
    
    ### Returns:
        np.ndarray[Tuple[bool, bool]]: A boolean array of shape (len(ref_graphs), len(test_graphs)) where each entry is True if the test graph is isomorphic to the corresponding reference graph.
    """
    pass

def test_isomorphism(ref:nx.Graph|List[nx.Graph], test:nx.Graph|List[nx.Graph])->np.ndarray[Tuple[bool, bool]] | np.ndarray[bool] | bool:
    refList:List[nx.Graph] = []
    ref_was_single = False
    testList:List[nx.Graph] = []
    test_was_single = False
    if isinstance(ref, nx.Graph):
        refList = [ref]
        ref_was_single = True
    else:
        refList = ref
    if isinstance(test, nx.Graph):
        testList = [test]
        test_was_single = True
    else:
        testList = test
    
    #now do the weisfeiler leman isomorphism test with color refinement
    def color_refinement(graphs:List[nx.Graph])->np.ndarray:
        """The color refinement algorithm for isomorphism testing.
        
        ### Args:
            graphs (List[nx.Graph]): The graphs to test for isomorphism.
            
        ### Returns:
            np.ndarray: The distributions of colors for each graph.
        """
        #TODO implement the color refinement algorithm
        pass

    ref_colors, test_colors = np.split(color_refinement(refList+testList), [len(refList)])
    
    #now compare the color distributions
    isomorphic = np.zeros((len(refList), len(testList)), dtype=bool)
    for i, ref_color in enumerate(ref_colors):
        for j, test_color in enumerate(test_colors):
            isomorphic[i, j] = np.all(ref_color==test_color)

    if ref_was_single and test_was_single:
        return isomorphic[0, 0]

    elif test_was_single:
        return isomorphic[:, 0]

    else:
        return isomorphic


