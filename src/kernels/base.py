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
import joblib as jl
import psutil

KerneledGraph = NewType("KerneledGraph", np.ndarray)
"""A type hint for the kerneled graph, which is a numpy array."""

class BaseKernel():

    def __init__(self, graph:nx.Graph, **kwargs):
        self.graph = graph
        ...

    @abc.abstractmethod
    def transform(self, **kwargs)->KerneledGraph:
        """the important function that implements the kerneling method and returns the kerneled graph."""
        raise NotImplementedError("The method transform must be implemented in a subclass.")

    @classmethod
    def readGraph(cls:"BaseKernel", graphPath:Path|str, index:int=0, **kwargs)->KerneledGraph:
        """A factory method that is initialized with some data and arbitrary arguments.
        As arguments, you may provide the init arguments of the class, and the transform arguments.    
        """
        graph: nx.Graph = None
        with open(graphPath if isinstance(graphPath, str) else graphPath.absolute(), "rb") as f:
            graph = pkl.load(f)[index]

        
        initKwargs = {k:v for k,v in kwargs.items() if k in inspect.signature(cls).parameters}
        transformKwargs = {k:v for k,v in kwargs.items() if k not in initKwargs}

        kernelObject:BaseKernel = cls(graph, initKwargs)
        
        return kernelObject.transform(graph, **transformKwargs)
    
    @classmethod
    def readGraphs(cls:"BaseKernel", graphPath:Path|str, **kwargs)->List[KerneledGraph]:
        """A factory method that is initialized with some data and arbitrary arguments.
        As arguments, you may provide the init arguments of the class, and the transform arguments.    
        """
        graphs:List[nx.Graph] = []
        with open(graphPath if isinstance(graphPath, str) else graphPath.absolute(), "rb") as f:
            graphs = pkl.load(f)

        
        initKwargs = {k:v for k,v in kwargs.items() if k in inspect.signature(cls).parameters}
        transformKwargs = {k:v for k,v in kwargs.items() if k not in initKwargs}

        # print(kwargs, initKwargs, transformKwargs) # debug-print
        kernelObjects:List[BaseKernel] = [cls(graph=graph, **initKwargs) for graph in graphs]
        
        return jl.Parallel(n_jobs=psutil.cpu_count())(jl.delayed(kernelObject.transform)(**transformKwargs) for kernelObject in kernelObjects)
    
    def show(self, saveOnly:bool=False, saveToPath:str|None=None):
        nx.draw_networkx(self.graph)

        if saveToPath:
            plt.savefig(saveToPath)
            if saveOnly:
                plt.close()
                return
        
        plt.show()

def dense_histogram(coloring:np.ndarray, colors:np.ndarray=None)->np.ndarray:
    """Compute the dense histogram of the colors.
    
    Args:
        coloring (np.ndarray): The coloring of the nodes. Dimensions: (n_graphs x) n_nodes
        
    Returns:
        np.ndarray: The dense histogram of the colors. Dimensions: (n_graphs x) n_bins (histogram)"""
    if len(coloring)==0:
        return np.array([0])

    single = False
    if len(coloring.shape) == 1:
        single = True
        coloring = coloring.reshape((1, coloring.shape[0]))

    mapping : Dict[int, int] = {color:i for i, color in enumerate(np.unique(coloring).tolist() if not colors else colors.tolist()) }#general mapping for all graphs to condense the colorings
        # print(mapping) #debug-print
    # print("Colors before mapping:", coloring) #debug-print    
    coloring = np.vectorize(mapping.get)(coloring) #densified
    if colors:
        colors = np.vectorize(mapping.get)(colors)
    # print("Colors after mapping:", coloring) #debug-print
    num_colors = np.max(coloring)+1 if not colors else np.max(colors)+1
    dense_histograms:np.ndarray = np.zeros((coloring.shape[0], num_colors), dtype=np.int8)
    #compute the histogram for each graph from the now condensed colorings
    for c in range(coloring.shape[0]):
        # print("Coloring:", coloring[c]) #debug-print
        # print("colorings[c]:", colorings[c], max_color+1) #debug-print
        dense_histograms[c, :num_colors] = np.histogram(
            coloring[c],
            bins=np.arange(num_colors+1), 
            density=False
        )[0] 
        # print("Dense histogram:", dense_histograms[c]) #debug-print

    if single:
        return dense_histograms[0]
    else:
        return dense_histograms

def new_color_hash(coloring:np.ndarray, graphs:List[nx.Graph], n_jobs=1)->np.ndarray[int, int]:
    """Hash the colors of the nodes in the graph. with info abou the current coloring and the adjacent nodes' colors.
    #Use the python hash function to hash the colors of the nodes in the graph, if we would save the colors in a dict we would hash them anyway so might as well do it directly.
    
    ### Args:
    coloring (np.ndarray): The current coloring of the nodes of multiple graphs. Dimensions: n_graphs x n_nodes.
    graphs (List[nx.Graph]): The graphs to hash the colors for.
    
    ### Returns:
    np.ndarray: The hashed colors of the nodes. Dimensions: n_graphs x n_nodes.
    """
    new_coloring = np.zeros_like(coloring, dtype=np.int64)
    seed = hash(str(coloring))
    num_colors = len(np.unique(coloring))
    # print(seed) #debug-print

    def update_graph(new_coloring, coloring, graph:nx.Graph, i:int, seed:int):
        #TODO: possibly parallelize this
        if graph.number_of_nodes() == 0:
            return
        nonzero_adj:Tuple[np.ndarray, np.ndarray] = nx.adjacency_matrix(graph).nonzero()

        # print(type(nonzero_adj[0]), type(nonzero_adj[1]), nonzero_adj[0], nonzero_adj[1])#debug-print
        for n, _ in enumerate(graph.nodes.keys()):
            neighbors:np.ndarray = nonzero_adj[1][np.where(nonzero_adj[0]==n)[0]]
            # print(f"Neighbors of node {n}: {neighbors}") #debug-print
            # print(f"Coloring of neighbors of node {n}: {coloring[i, neighbors]}") #debug-print
            # print(f"Histogram of neighbors of node {n}: {dense_histogram(coloring[i, neighbors])}") #debug-print
            hash_input = np.concatenate((
                dense_histogram(coloring[i, neighbors], colors=np.unique(coloring[i])),
                [coloring[i, n]]
            ))
            # print(f"Hashinput for node {n}: {hash_input}") #debug-print
            new_coloring[i, n] = xxhash.xxh32_intdigest(hash_input.tobytes(), seed=seed)
            # print(f"New color for node {n}: {new_coloring[i, n]}") #debug-print
            del hash_input

    #parallelize this if jobs!=1
    if n_jobs!=1:
        jl.Parallel(n_jobs=n_jobs if n_jobs>0 else psutil.cpu_count())(jl.delayed(update_graph)(new_coloring, coloring, graph, i, seed) for i, graph in enumerate(graphs))
    else:
        for i, graph in enumerate(graphs):
            update_graph(new_coloring, coloring, graph, i, seed)

    # print(np.where(new_coloring==np.nan)) #debug-print
    return new_coloring



#now do the weisfeiler leman isomorphism test with color refinement
@overload
def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:Literal[False]=False, return_dense:Literal[False]=False)->sparseFun.csr_array:
    """The color refinement algorithm for isomorphism testing.
    
    ### Args:
        graphs (List[nx.Graph]): The graphs to test for isomorphism.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_iterations (int): How many iterations to perform, if negative until the refinement is stable. Currently only positive values are supported.
        returnAllIterations==False:  Whether to return all iterations. Return only the last iteration
        return_dense==False: Whether to return the dense histogram instead of the sparse one.
        
    ### Returns:
        scipy.sparse.csr_array: The distributions of colors for each graph. dimensions: n_graphs x n_bins (histogram)
    """

    pass

@overload
def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:Literal[False]=False, return_dense:Literal[True]=True, n_jobs=1)->np.ndarray:
    """The color refinement algorithm for isomorphism testing.
    
    ### Args:
        graphs (List[nx.Graph]): The graphs to test for isomorphism.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_iterations (int): How many iterations to perform, if negative until the refinement is stable. Currently only positive values are supported.
        returnAllIterations==False:  Whether to return all iterations. Return only the last iteration
        return_dense==True: Whether to return the dense histogram instead of the sparse one.
        
    ### Returns:
        numpy.ndarray: The distributions of colors for each graph. dimensions: n_graphs x n_bins (histogram)
    """

    pass


@overload
def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:Literal[True]=True, return_dense:Literal[False]=False, n_jobs=1)->List[sparseFun.csr_array]:
    """The color refinement algorithm for isomorphism testing.
    
    ### Args:
        graphs (List[nx.Graph]): The graphs to test for isomorphism.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_iterations (int): How many iterations to perform, if negative until the refinement is stable. Currently only positive values are supported.
        returnAllIterations==True:  Whether to return all iterations. Uses a list of sparse arrays.
        return_dense==False : Whether to return the dense histogram instead of the sparse one.
        
    ### Returns:
        List[scipy.sparse.csr_array]: The distributions of colors for each graph. dimensions: (n_iterations x) n_graphs x n_bins (histogram)
    """
    pass

@overload
def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:Literal[True]=True, return_dense:Literal[True]=True, n_jobs=1)->List[np.ndarray]:
    """The color refinement algorithm for isomorphism testing.
    
    ### Args:
        graphs (List[nx.Graph]): The graphs to test for isomorphism.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_iterations (int): How many iterations to perform, if negative until the refinement is stable. Currently only positive values are supported.
        returnAllIterations==True:  Whether to return all iterations. Uses a list of sparse arrays.
        return_dense==True : Whether to return the dense histogram instead of the sparse one.
        
    ### Returns:
        List[numpy.ndarray]: The distributions of colors for each graph. dimensions: (n_iterations x) n_graphs x n_bins (histogram)
    """
    pass

def color_refinement(graphs:List[nx.Graph], useLabels:bool=False, n_iterations:int=-1, returnAllIterations:bool=False, return_dense:bool=False, n_jobs=1)->List[sparseFun.csr_array] | List[np.ndarray] | sparseFun.csr_array | np.ndarray:

    coloring_stable:bool=False
    iteration:int = 0
    def init_coloring(graphs:List[nx.Graph], useLabels:bool=False)->np.ndarray[np.int64]:
        """Initialize the colorings of the nodes.
        
        Args: 
            graphs (List[nx.Graph]): The graphs to initialize the colorings for.
            useLabels (bool): Whether to use the node labels to initialize the colorings.
            
        Returns:
            np.ndarray: The initialized colorings. Dimensions: n_graphs x n_nodes
        """
        # print("init_coloring") #debug-print
        # print(type(graphs), type(graphs[0]))
        #init the colorings with 5 iterations, if we need more, we will realloc later
        #the colorings holds the color for each node, we later aggregate into sparse histogram representation
        colorings:np.ndarray[np.int64] = np.zeros((5 if n_iterations<0 else n_iterations, len(graphs), max(graph.number_of_nodes() for graph in graphs)), dtype=np.int64)
         
        if useLabels:
            for i, graph in enumerate(graphs):
                colorings[0][i][:graph.number_of_nodes()] = np.array([
                    int(nv) if nv!=None else -1 for nv in dict(
                        graph.nodes(data="node_label")
                    ).values()
                ])
                #fill nan with zeros
                # colorings[0][i] = np.nan_to_num(colorings[0][i])
        else:
            for g, graph in enumerate(graphs):
                # print("init_coloring graph", g) #debug-print
                colorings[0][g][:graph.number_of_nodes()] = np.ones((graph.number_of_nodes(),))

        return colorings

    colorings:np.ndarray[np.int64] = init_coloring(graphs, useLabels)#n_iterations x n_graphs x n_nodes
    # print(colorings[0]) #debug-print
    # print(np.where(colorings==np.nan)) #debug-print
    # print(last_iteration)
    if n_iterations < 0:
        while(not coloring_stable):
            # print(f"iteration {iteration}")
            iteration += 1
            if iteration >= colorings.shape[0]: #realloc if we need more iterations
                colorings = np.concatenate([colorings, np.zeros((1, len(graphs), max(graph.number_of_nodes() for graph in graphs)))], axis=0)
            colorings[iteration] = new_color_hash(colorings[iteration-1], graphs, n_jobs=n_jobs)
            

            #check if the coloring is stable
            #try to map the colors to the previous iteration
            #if the mapping is bijective, the coloring is stable
            #TODO implement the stable coloring check
            if n_iterations < 0:
                raise NotImplementedError("The stable coloring check is not implemented yet.")

    else:
        while(n_iterations>0 and iteration < n_iterations-1):
            # print(f"iteration {iteration}")
            iteration += 1
            if iteration >= colorings.shape[0]:
                colorings = np.concatenate([colorings, np.zeros((1, len(graphs), max(graph.number_of_nodes() for graph in graphs)))], axis=0)
            colorings[iteration] = new_color_hash(colorings[iteration-1], graphs)
            # if len(np.where(new_coloring==np.nan))>0: print("nan in colorings:", np.where(new_coloring==np.nan)) #debug-print
            
    
    def collectColoring(colorings:np.ndarray, return_dense:Literal[False]=False)->sparseFun.csr_array:
        """Collect the coloring into a sparse histogram representation.
        
        Args: 
            colorings (np.ndarray): The coloring of the nodes. Dimensions: n_graphs x n_nodes
            return_dense == False: Whether to return the dense histogram instead of the sparse one.
            
        Returns:
            sparseFun.csr_array: The sparse histogram of the colors. Dimensions: n_graphs x n_bins (histogram)
        """
        pass
    def collectColoring(colorings:np.ndarray, return_dense:Literal[True]=True)->np.ndarray:
        """Collect the coloring into a dense histogram representation. The colors are mapped into a dense space, after which the histogram is computed.
        
        Args: 
            colorings (np.ndarray): The coloring of the nodes. Dimensions: n_graphs x n_nodes
            return_dense == True: Whether to return the dense histogram instead of the sparse one.
            
        Returns:
            numpy.ndarray: The dense histogram of the colors. Dimensions: n_graphs x n_bins (histogram)
        """
        pass
    

    def collectColoring(colorings:np.ndarray, return_dense:bool=False)->sparseFun.csr_array | np.ndarray:
        """Collect the coloring into a sparse histogram representation.
        
        Args: 
            colorings (np.ndarray): The coloring of the nodes. Dimensions: n_graphs x n_nodes
            return_dense (bool): Whether to return the dense histogram instead of the sparse one.
            
        Returns:
            sparseFun.csr_array | numpy.ndarray: The sparse/dense histogram of the colors. Dimensions: n_graphs x n_bins (histogram)
        """
        # # print("collecting colorings:", colorings.shape, colorings) #debug-print
        # #condense the colorings into a non-sparse numpy histogram first, then spread it into a sparse scipy array
        # # graph_mappings:List[Dict[int, int]] = [{
        # #         color:i for i, color in enumerate(np.unique(colorings[c]).tolist())
        # #     } for c in range(colorings.shape[0])
        # # ] #mapping per graph because we need them later to uncondense the histograms
        # mapping : Dict[int, int] = {color:i for i, color in enumerate(np.unique(colorings).tolist()) }#general mapping for all graphs to condense the colorings
        # # print(mapping) #debug-print

        # colorings = np.vectorize(mapping.get)(colorings) #densified
        # max_color = np.max(colorings)
        # dense_histograms:np.ndarray = np.zeros((colorings.shape[0], max_color+1), dtype=np.int8)
        # #compute the histogram for each graph from the now condensed colorings
        # for c in range(colorings.shape[0]):
        #     # print("colorings[c]:", colorings[c], max_color+1) #debug-print
        #     dense_histograms[c, :max_color] = np.histogram(
        #         colorings[c],
        #         bins=np.arange(max_color+1), 
        #         density=False
        #     )[0] 
        # #REDO: maybe we can use the dense colorings directly, instead of the sparse histograms, because we just throw it into the hash function again. thus is is not necessary to uncondense it. #APPLIED

        ## ^^^^ outsourced into dense_histogram ^^^^


        data = dense_histogram(colorings)
        # print(data.shape, len(graphs), max(mapping.keys())+1)
        # print("data:", data)
        if return_dense:
            return data

        else:#
            return sparseFun.csr_matrix(data)

        #previous back mapping into csr with the inverse mapping, but makes no sense, so might as well not do it.

        # rows:np.ndarray = np.concatenate([
        #     np.full(
        #         shape=dense_histograms[i].shape, 
        #         fill_value=i
        #     ) \
        #     for i in range(len(dense_histograms))
        # ], axis=0)
        # #the columns are the bins of the histograms, these are uncondensed by the mapping
        # cols:np.ndarray = np.concatenate([
        #     np.array(list(graph_mappings[i].keys())) \
        #     for i in range(len(dense_histograms))
        # ], axis=0)

        # return sparseFun.csr_array(
        #     (data, (rows, cols)), shape=(colorings.shape[0], max(mapping.keys())+1)
        # )


    if returnAllIterations:
        return [collectColoring(colorings[i], return_dense=return_dense) for i in range(iteration+1)]#usually == n_iterations
    else:
        return collectColoring(colorings[iteration], return_dense=return_dense)


@overload
def test_isomorphism(ref_graph:nx.Graph, test_graph:nx.Graph, k:int=5, useLabels:bool=False, n_jobs:int=1)->bool:
    """Test the isomorphism of the test graph against the reference graph.
    
    ### Args:
        ref_graph (nx.Graph): The reference graph.
        test_graph (nx.Graph): The test graph.
        k (int): The number of color refinement iterations to perform.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_jobs (int): The number of jobs to parallelize the color refinement.

    ### Returns:
        bool: True if the graphs are isomorphic, False otherwise.
    """
    pass

@overload
def test_isomorphism(ref_graphs:List[nx.Graph], test_graph:nx.Graph, k:int=5, useLabels:bool=False, n_jobs:int=1)->np.ndarray[bool]:
    """Test the isomorphism of the test graphs against the reference graphs.
    
    ### Args:
        ref_graphs (List[nx.Graph]): The reference graphs.
        test_graph (nx.Graph): The graph to test to what reference graphs it is isomorphic.
        k (int): The number of color refinement iterations to perform.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_jobs (int): The number of jobs to parallelize the color refinement.
    
    ### Returns:
        np.ndarray[bool]: A boolean array of length len(ref_graphs) where each entry is True if the test graph is isomorphic to the corresponding reference graph.
    """
    pass

@overload
def test_isomorphism(ref_graphs:List[nx.Graph], test_graphs:List[nx.Graph], k:int=5, useLabels:bool=False, n_jobs:int=1)->np.ndarray[bool, bool]:
    """Test the isomorphism of the test graphs against the reference graphs.
    
    ### Args:
        ref_graphs (List[nx.Graph]): The reference graphs.
        test_graphs (List[nx.Graph]): The graphs to test for isomorphism.
        k (int): The number of color refinement iterations to perform.
        useLabels (bool): Whether to distinguish between nodes based on node-labels.
        n_jobs (int): The number of jobs to parallelize the color refinement.
    
    ### Returns:
        np.ndarray[Tuple[bool, bool]]: A boolean array of shape (len(ref_graphs), len(test_graphs)) where each entry is True if the test graph is isomorphic to the corresponding reference graph.
    """
    pass

def test_isomorphism(ref:nx.Graph|List[nx.Graph], test:nx.Graph|List[nx.Graph], k:int=5, useLabels:bool=False, n_jobs:int=1)->np.ndarray[bool, bool] | np.ndarray[bool] | bool:
    refList:List[nx.Graph] = []
    ref_was_single = False
    testList:List[nx.Graph] = []
    test_was_single = False
    
    # print(type(ref), type(test)) #debug-print


    if isinstance(ref, list):
        refList = ref
    else:
        refList = [ref]
        ref_was_single = True

    if isinstance(test, list):
        testList = test
    else:
        testList = [test]
        test_was_single = True

    # print(type(refList), type(testList)) #debug-print

    colors = color_refinement(refList+testList, n_iterations=k, useLabels=useLabels, returnAllIterations=False, n_jobs=n_jobs)
    # print(colors.shape)
    ref_colors, test_colors = colors[:len(refList)], colors[len(refList):]
    # print(ref_colors.shape, test_colors.shape)
    #now compare the color distributions
    isomorphic = np.zeros((len(refList), len(testList)), dtype=bool)
    for i in range(len(refList)):
        for j in range(len(testList)):
            isomorphic[i,j] = 1 - np.any(ref_colors.toarray()[i]!=test_colors.toarray()[j])

    if ref_was_single and test_was_single:
        return isomorphic[0, 0]

    elif test_was_single:
        return isomorphic[:, 0]

    else:
        return isomorphic


