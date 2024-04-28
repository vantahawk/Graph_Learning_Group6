import abc
import networkx as nx
import pickle as pkl
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import NewType, List, overload, Tuple, Literal, Dict
import scipy.sparse as sparseFun
import scipy.stats
import xxhash
import scipy

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


def new_color_hash(coloring:np.ndarray, graphs:List[nx.Graph])->np.ndarray:
    """Hash the colors of the nodes in the graph. with info abou the current coloring and the adjacent nodes' colors.
    
    ### Args:
    coloring (np.ndarray): The current coloring of the nodes of multiple graphs. Dimensions: n_graphs x n_nodes.
    graphs (List[nx.Graph]): The graphs to hash the colors for.
    
    ### Returns:
    np.ndarray: The hashed colors of the nodes. Dimensions: n_graphs x n_nodes.
    """
    new_coloring = np.zeros_like(coloring)
    #
    hash = xxhash.xxh32(seed=hash(str(coloring)))

    for i, graph in enumerate(graphs):
        nonzero_adj:Tuple[np.ndarray, np.ndarray] = nx.adjacency_matrix(graph).nonzero()

        for node in enumerate(graph.nodes.keys()):
            neighbors:np.ndarray = nonzero_adj[1,np.where(nonzero_adj[0]==node)[0]]

            hash.update(coloring[i, [node]+neighbors.tolist()])
            new_coloring[i, node] = hash.intdigest()

    return new_coloring

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
    @overload
    def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:Literal[False]=False)->sparseFun.csc_array:
        """The color refinement algorithm for isomorphism testing.
        
        ### Args:
            graphs (List[nx.Graph]): The graphs to test for isomorphism.
            useLabels (bool): Whether to distinguish between nodes based on node-labels.
            n_iterations (int): How many iterations to perform, if negative until the refinement is stable. Currently only positive values are supported.
            returnAllIterations==False:  Whether to return all iterations. Return only the last iteration
            
        ### Returns:
            scipy.sparse.csr_array: The distributions of colors for each graph. dimensions: n_graphs x n_bins (histogram)
        """

        pass

    @overload
    def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:Literal[True]=True)->List[sparseFun.csr_array]:
        """The color refinement algorithm for isomorphism testing.
        
        ### Args:
            graphs (List[nx.Graph]): The graphs to test for isomorphism.
            useLabels (bool): Whether to distinguish between nodes based on node-labels.
            n_iterations (int): How many iterations to perform, if negative until the refinement is stable. Currently only positive values are supported.
            returnAllIterations==True:  Whether to return all iterations. Uses a list of sparse arrays.
            
        ### Returns:
            List[scipy.sparse.csr_array]: The distributions of colors for each graph. dimensions: (n_iterations x) n_graphs x n_bins (histogram)
        """
        pass

    def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:bool=False)->List[sparseFun.csc_array] | sparseFun.csc_array:

        coloring_stable:bool=False
        iteration:int = 0
        #init the colorings with 5 iterations, if we need more, we will realloc later
        #the colorings holds the color for each node, we later aggregate into sparse histogram representation
        colorings:np.ndarray = np.ones((5, len(graphs), max(graph.number_of_nodes() for graph in graphs)))
        if useLabels:
            for i, graph in enumerate(graphs):
                colorings[0][i] = np.array([
                    nv for nv in dict(
                        graph.nodes(data="label")
                    ).values()
                ])
        while(not coloring_stable or (n_iterations>0 and iteration < n_iterations)):
            colorings[iteration+1] = new_color_hash(colorings[iteration], graphs)
            iteration += 1

            #check if the coloring is stable
            #try to map the colors to the previous iteration
            #if the mapping is bijective, the coloring is stable
            #TODO implement the stable coloring check
            if n_iterations < 0:
                raise NotImplementedError("The stable coloring check is not implemented yet.")

        def collectColoring(colorings:np.ndarray)->sparseFun.csr_array:
            """Collect the coloring into a sparse histogram representation.
            
            Args: 
                colorings (np.ndarray): The coloring of the nodes. Dimensions: n_graphs x n_nodes
                
            Returns:
                sparseFun.csr_array: The sparse histogram of the colors. Dimensions: n_graphs x n_bins (histogram)"""
            #condense the colorings into a non-sparse numpy histogram first, then spread it into a sparse scipy array
            graph_mappings:List[Dict[int, int]] = [
                {color:i for i, color in enumerate(np.unique(colorings[c]).tolist())
                } for c in range(colorings.shape[0])
            ] #mapping per graph because we need them later to uncondense the histograms
            mapping = Dict[int, int] = {color:i for i, color in enumerate(np.unique(colorings).tolist())
            }#general mapping for all graphs to condense the colorings
            colorings = np.vectorize(mapping.get)(colorings)
            #compute the histogram for each graph from the now condensed colorings
            dense_histograms:List[np.ndarray] = [
                np.histogram(
                    colorings[c],
                    bins=np.arange(np.max(colorings[c])+1), 
                    density=False#
                )[0] 
                for c in range(colorings.shape[0])
            ] 
            #REDO: maybe we can use the dense colorings directly, instead of the sparse histograms, because we just throw it into the hash function again. thus is is not necessary to uncondense it.

            data:np.ndarray = np.concatenate(dense_histograms, axis=0)
            rows:np.ndarray = np.concatenate([
                np.full(
                    shape=dense_histograms[i].shape, 
                    fill_value=i
                ) \
                for i in range(len(dense_histograms))
            ], axis=0)
            #the columns are the bins of the histograms, these are uncondensed by the mapping
            cols:np.ndarray = np.concatenate([
                np.array(list(graph_mappings[i].keys())) \
                for i in range(len(dense_histograms))
            ], axis=0)

            return sparseFun.csr_array(
                (data, (rows, cols)), shape=(colorings.shape[0], max(mapping.keys())+1)
            )


        if returnAllIterations:
            return [collectColoring(coloring) for coloring in colorings[:iteration]]#usually == n_iterations
        else:
            return collectColoring(colorings[iteration])

    colors = color_refinement(refList+testList, n_iterations=5)
    ref_colors, test_colors = colors[:len(refList)], colors[len(refList):]
    
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


