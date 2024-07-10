import networkx as nx
import numpy as np
import torch as th
from torch.utils.data import Dataset
import yaml
from joblib import Parallel, delayed
import random
from .geomConvert import zmat_from_molecule
from scipy.spatial.distance import cdist

feature_config_path="src/feature_config.yaml"

def one_hot_encoder(label: int, length: int) -> np.ndarray:
    '''returns one-hot vector according to given label integer and length'''
    return np.eye(length)[label]

def get_distances(graph:nx.Graph) -> np.ndarray:
    """Expects a molecular graph with xyz coordinates as node attributes
    
    Args:
    - graph (nx.Graph): input graph
    
    Returns:
    (np.ndarray): pairwise distances between nodes
    """
    xyz_array = np.array([node[1] for node in graph.nodes(data="node_attributes")])

    #distances
    return cdist(xyz_array, xyz_array)

def calculate_hosoya_index(graph:nx.Graph, node:int, dist:int, size:int, seed:int=0)->int:
    """Calculates the hosoya index for a graph subset, with bfs starting at node, up to a certain distance and size."""
    def bfs(graph:nx.Graph, start:int, dist:int, size:int, seed:int)->list[int]:
        random.seed(seed)

        visited = [False]*graph.number_of_nodes()
        queue = [start]
        visited[start] = True
        level = 0
        while level < dist and sum(visited) < size:
            new_queue = []
            for q in queue:
                neighbors = list(graph.neighbors(q))
                random.shuffle(neighbors)
                for n in neighbors:
                    if not visited[n]:
                        visited[n] = True
                        new_queue.append(n)
            queue = new_queue
            level += 1
            
        return visited

    def hosoya(graph:nx.Graph, nodes_bool:list[int])->int:
        """Calculates the hosoya index for a graph subset."""
        #get the subgraph
        nodes = np.arange(graph.number_of_nodes())[nodes_bool]
        subgraph = graph.subgraph(nodes)
        return nx.maximal_matching(subgraph)

    sub = bfs(graph, node, dist, size, seed)
    return len(hosoya(graph, sub))


def cwk_node_contributions(graph:nx.Graph, max_length:int=5, array_size:int|None=None, cut_duplicates:bool=True)->np.ndarray:
    """
    Compute the contribution of each node to the closed walks up to given length in the graph.
    
    Args:
    - graph (nx.Graph): input graph
    - length (int): int, maximum length of walks to consider
    - array_size (int): Could be used to inform the function about the maximum number of nodes in the graph.
    
    Returns:
    - node_scores: numpy array, scores for each node based on their contribution to closed walks.
    """
    # Compute eigenvalues and eigenvectors of the adjacency matrix
    A = nx.adjacency_matrix(graph).todense()
    eigvals, eigvecs = np.linalg.eig(A)
    node_count = graph.number_of_nodes()
    # Initialize an array to store the scores for each node
    node_scores = np.zeros((node_count if array_size == None else array_size, max_length))
    
    # Compute contributions for each node
    for k in range(max_length):
        # do this in parallel using numpy
        node_scores[:node_count, k] = np.sum((eigvecs ** 2) * (eigvals ** k), axis=1)

    #round to integer (make low nonzero values e-14 disappear)
    node_scores = np.floor(node_scores).astype(int)

    if cut_duplicates:
        #remove duplicates, meaning e.g. for a length 12, remove all the subfactor influences
        for i in range(2, max_length//2+1):
            for j in range(2*i, max_length, i):
                node_scores[:, j] -= node_scores[:, i]

        #quite interesting in how the duplicates are removed:
        # 1. for each possible length that has an influence on another length
        # 2. for each node that it has an influence on
        # 3. remove the influence of the first length on the second length
        # that way, we remove only the influences that really are there, if done in other order, we would remove influences twice

    return node_scores

def valence_electrons(a:int, s:int)->np.ndarray:
    """Returns the number of valence electrons of the atom with number a, up to shell number s"""
    #shell sizes
    # total_shell_sizes = [2, 8, 8, 18, 18, 32, 32]
    # shell_sizes = {
    #     "s": 2,
    #     "p":6,
    #     "d":10,
    #     "f":14
    # }
    ve = np.zeros((s,))
    if a <= 2:
        ve[0] = a
    elif a <= 10:
        a -=2
        ve[0] = 1 if a==1 else 2
        ve[1] = max(0, a-2)
    elif a <= 18:
        a-=10
        ve[0] = 1 if a==1 else 2
        ve[1] = max(0, a-2)
    elif a <= 36:
        a-=18
        ve[0] = 1 if a==1 else 2
        ve[2] = max(0, a-2) #first the d shell gets filled
        ve[1] = max(0, a-12)
    else:
        raise ValueError("Currently we only support atoms up to count 36")

    return ve


def make_node_features(graphs:list[nx.Graph], dist_dim:int)-> list[np.ndarray]:
    """Makes from a list of graphs a list of node feature for each graph."""
    
    with open(feature_config_path, "r") as f:
        feature_config = yaml.safe_load(f)

    """ENCODE THE NODE LABELS: ATOM TYPES"""
    #we have the node labels, that must be one-hot encoded: [1..35]
    node_labels = [np.zeros((graph.number_of_nodes(), 35)) for graph in graphs]
    for g in range(len(node_labels)):
        for n, nname in enumerate(graphs[g].nodes(data="node_label")):
            node_labels[g][n, nname[1]-1] = 1


    """THE NUMBER OF BONDS FOR EACH ATOM, AND THE NUMBER OF VALENCE ELECTRONS in each shell"""
    electrons = [np.zeros((graph.number_of_nodes(), 4)) for graph in graphs]
    for g in range(len(electrons)):
        for n, nname in enumerate(graphs[g].nodes(data="node_label")):
            electrons[g][n][0] = graphs[g].degree(nname[0])
            electrons[g][n][1:4] = valence_electrons(nname[1], 3)

    """MAKE ZMATRIX PER GRAPH"""
    #we have 3d vectors for each atom, that must be converted to zmatrix
    #question: how to handle the missing values in the zmatrix at the beginning? set to 0 to not perturb the normalization
    zmatrices = [np.zeros((graph.number_of_nodes(), 7)) for graph in graphs]
    
    #fill in the shape if dome molecules are smaller
    # make_zmatrix = decorate_shape((max_node_count, 7))(zmat_from_molecule)
    # zmatrices = np.array(Parallel(n_jobs=-1)(delayed(make_zmatrix)(graph) for graph in graphs))
    for g in range(len(zmatrices)):
        zmatrices[g] = zmat_from_molecule(graphs[g])

    """COMPUTE THE DISTANCES FROM NODE TO NODE, THIS IS HIGHLY INFORMATIVE"""

    distances = [np.zeros((graph.number_of_nodes(), dist_dim)) for graph in graphs]
    for g,graph in enumerate(graphs):
        distances[g][:,:graph.number_of_nodes()] = get_distances(graph)

    """MAKE HELPFUL ATOM FEATURES"""
    #further load the helpful features for each atom: electro-negativities and the first 4 ionization energies, and 0th energy(affinity)
    with open("helpful_extra_features/ionizationenergies.yaml", "r") as f:
        ionization_energies = yaml.load(f, Loader=yaml.FullLoader)

    with open("helpful_extra_features/electronegativities.yaml", "r") as f:
        electronegativities = yaml.load(f, Loader=yaml.FullLoader)
    
    with open("helpful_extra_features/electronaffinities.yaml", "r") as f:
        electronaffinities = yaml.load(f, Loader=yaml.FullLoader)

    help_features = [np.zeros((graph.number_of_nodes(), 6)) for graph in graphs]
    for g in range(len(help_features)):
        for n, nname in enumerate(graphs[g].nodes(data="node_label")):
            atom_type = nname[1]
            help_features[g][n] = np.array([
                electronegativities[atom_type], 
                electronaffinities[atom_type], 
                *ionization_energies[atom_type]
            ])

    """BASICALLY A CLOSED WALK KERNEL BUT WITH NOT CONTRIBUTIONS: CIRCLES EACH NODE CONTRIBUTES TO"""
    #further do circle finding, see ex. 1: important for benzene rings and similar structures.
    #do this up to 12 atoms, longer circles are not that important in small molecules
    circles = [np.zeros((graph.number_of_nodes(), feature_config["circle"]["length"])) for graph in graphs]

    circles = Parallel(n_jobs=-1)(
        delayed(cwk_node_contributions)(
            graph=graph, 
            max_length=feature_config["circle"]["length"],
            cut_duplicates=feature_config["circle"]["cut_duplicates"]
        ) for graph in graphs
    )

    """BASICALLY A HOSOYA INDEX KERNEL ON RANDOM NODE CENTERED SUBGRAPHS"""
    #maybe do hosoya index calculations for fixed-size subsets of the graph
    #e.g. do a bfs from each node up to a certain length/size found, then calculate hosoya index, and attribute it to the node
    #use min_distance=2, min_size=5, whenever one is reached stop
    hosoya_indexes = [np.zeros((graph.number_of_nodes(),feature_config["hosoya"]["num_samples"]+1)) for graph in graphs]

    for g in range(len(hosoya_indexes)):
        for n in range(graphs[g].number_of_nodes()):
            for sample in range(feature_config["hosoya"]["num_samples"]):
                hosoya_indexes[g][n, sample] = calculate_hosoya_index(
                    graph=graphs[g], 
                    node=n, 
                    dist=feature_config["hosoya"]["depth"], 
                    size=feature_config["hosoya"]["size"],
                    seed=sample
                )
    #add the whole graph hosoya index as well
    for g in range(len(hosoya_indexes)):
        for n in range(hosoya_indexes[g].shape[0]):
            hosoya_indexes[g][n, -1] = len(nx.maximal_matching(graphs[g]))

    #of course normalize everything to [0,1], that is not already normalized (e.g. the one-hot vectors)
    #normalizing the zmatrices by global min, max, because otherwise it might vary too much, this should still be helpful.
    zmatrix_global_max = max([zmatrix.max() for zmatrix in zmatrices])
    zmtrix_global_min = min([zmatrix.min() for zmatrix in zmatrices])
    zmatrices = [(zmatrix - zmtrix_global_min) for zmatrix in zmatrices]
    zmatrices = [zmatrix/zmatrix_global_max for zmatrix in zmatrices]#zmatrices / zmatrices.max()

    # distances, electrons, circles, hosoya_indexes, help_features all should not be normalized, as they could vary drastically between molecules and we want to keep that
    electrons = [electron/10 for electron in electrons] #10 is max fill of d shell

    #concatenate all features
    node_features = [np.concatenate([node_labels[g], zmatrices[g], distances[g], help_features[g], circles[g], hosoya_indexes[g]], axis=1) for g,_ in enumerate(graphs)]
    return node_features

def make_edge_features(graphs:list[nx.Graph])->list[np.ndarray]:
    """Makes from a list of graphs a list of edge features"""
    
    edge_features = [ 
        np.array([one_hot_encoder(edge[2]['edge_label'] - 1, 5) for _ in range(2)
            for edge in graph.edges(data=True) 
        ]) for graph in graphs
    ] #onehot encoded edge labels, with one extra column, for the distance

    dists = [get_distances(graph) for graph in graphs]
    
    def get_node_index(graph:nx.Graph, edge:int)->int:
        return list(graph.nodes()).index(edge)

    #only use those distances that are actually there
    for g, graph in enumerate(graphs):
        for e, edge in enumerate(graph.edges()):
            for i in range(2):
                edge_features[g][2*e+i, -1] = dists[g][get_node_index(graph, edge[0]), get_node_index(graph, edge[1])]

    return edge_features

class Custom_Dataset(Dataset):
    '''implements custom [Dataset] class for given dataset (list of nx.graphs)'''
    def __init__(self, graphs: list[nx.Graph], is_test:bool=False, size:int|None = None, node_features_size=None, seed:int=0) -> None:
        """Make the custom graphs dataset that can be collated

        Args:
        - graphs (list[networkx.Graph]): the graphs to make into a dataset
        - is_test(bool) : Whether to fill in the labels with blank
        - size (int): The number of graphs to use, if None, use all
        - node_features_size: A shape that the node features per node must match. This likely means its an integer
        - seed (int): The seed to use for shuffling the graphs
        """
        super().__init__()

        # optional attributes
        self.graphs = graphs
        random.seed(seed)
        random.shuffle(self.graphs)
        if size is not None:
            self.graphs = self.graphs[:size]

        self.length = len(graphs)
        self.sizes = [nx.number_of_nodes(graph) for graph in graphs]

        # pre-compute sparse representation for all graphs in dataset
        self.nodes_start = [[edge[0] for edge in graph.edges(data=True)]
                            for graph in graphs]  # start nodes all edges in graph
        self.nodes_end = [[edge[1] for edge in graph.edges(data=True)]
                          for graph in graphs]  # end nodes all edges in graph
        self.edge_idx = [th.tensor(np.array([self.nodes_start[index] + self.nodes_end[index],
                                             self.nodes_end[index] + self.nodes_start[index]]), dtype=th.long)
                         for index in range(self.length)]  # directed edge index list, i.e. w/ reversed duplicate

        #because we include distances in the node feature, this dimension is fixed to the maximum node count
        self.node_features = make_node_features(graphs, node_features_size) #for graph in graphs for node in graph.nodes()
        self.node_features = [th.tensor(self.node_features[g]) for g,_ in enumerate(graphs)]

        self.edge_features = make_edge_features(graphs) #for graph in graphs
        self.edge_features = [th.tensor(self.edge_features[g]) for g,_ in enumerate(graphs)]
        
        self.graph_labels = [th.tensor(np.array(graph.graph['label'] if not is_test else -1), dtype=th.float) for graph in graphs]  # scalar, real-valued


    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        '''returns custom sparse representation for each graph by index as tuple of th.tensors
        These are the edge_indexes, the node_features, the edge_features, and the graph_label
        '''
        #return super().__getitem__(index)

        # draw sparse representation from constructor for each graph index
        return self.edge_idx[index], self.node_features[index], self.edge_features[index], self.graph_labels[index]


    def __len__(self) -> int:
        '''length of dataset/batch, i.e. number of graphs in it'''
        return len(self.graphs)
