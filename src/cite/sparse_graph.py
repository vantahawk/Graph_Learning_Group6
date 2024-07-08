'''implememtation for sparse representaion of graph (origially intended for CITE)'''
#import networkx as nx
from networkx import DiGraph, MultiDiGraph, Graph, adjacency_matrix, is_directed, relabel_nodes
#import numpy as np
from numpy import array, sum, zeros
#import torch as th
from torch import float, long, tensor
from torch.utils.data import Dataset


"""
def one_hot_encoder(label: int, length: int) -> ndarray:
    '''returns one-hot vector according to given label integer and length'''
    zero_vector = zeros(length)
    #if label != None:  # maintain zero vector for empty labels
    #    zero_vector[label] = 1
    zero_vector[label] = 1
    return zero_vector
"""

"""
def remap(edge_nodes: list[int], node_idx: list[int]) -> list[int]:
    '''remaps start & end node lists from edges directly using node_idx'''
    for i in range(len(edge_nodes)):
        edge_nodes[i] = node_idx.index(edge_nodes[i])
    return edge_nodes
"""


#class Sparse_Graph(Dataset):  # child of Dataset class, decomment to use in DataLoader
class Sparse_Graph():  # generic class
    '''implements sparse representation of given, *single* nx.[graph], originally for Graph (like CITE), partially adapted for DiGraph & MulitDiGraph (like LINK) as well, would still need to deal w/ "id" of edges in LINK'''
    def __init__(self, graph: Graph | DiGraph | MultiDiGraph, set_node_labels: bool, set_edge_labels: bool = False
                 #, n_node_labels: int = 4  # 4 node label classes in CITE
                 ) -> None:
        #super().__init__()

        self.n_nodes = graph.number_of_nodes()
        #self.n_edges = graph.number_of_edges()
        self.node_idx = [node[0] for node in graph.nodes(data=True)]  # list of node indices
        # node index remapping:
        self.node_map = {self.node_idx[i] : i for i in range(self.n_nodes)}  # node remapping dictionary w/ elem.s of form (old_idx: new_idx)
        graph = relabel_nodes(graph, self.node_map, copy=True)  # equiv. [graph] but w/ remapped node indices
        #
        # key attributes:
        self.set_node_labels = set_node_labels
        self.nodes = graph.nodes(data=True)
        #self.node_idx = [node[0] for node in self.nodes]  # list of node indices
        self.edges = graph.edges(data=True)
        #self.first_node = min(self.node_idx)  # account for node count via 1st node index
        self.node_idx_remap = list(range(self.n_nodes))  # remapped node indices = [0,...,n_nodes-1] (order not important here)

        # pre-compute sparse representations of given [graph]:
        self.adj_mat = adjacency_matrix(graph)#.toarray()
        # for degree normalization like in GCN but for using scatter:
        self.degree_factors = tensor((sum(self.adj_mat.toarray(), axis=-1) + 1) ** -0.5).reshape((self.n_nodes, 1))
        self.nodes_start = [edge[0] for edge in self.edges]  # start nodes all edges in graph
        self.nodes_end = [edge[1] for edge in self.edges]  # end nodes all edges in graph
        # directed edge index list w/ added self-loops for self-message-passing:
        if is_directed(graph):  # for (multi-)directed [graph] like LINK
            self.edge_idx = tensor(array([self.nodes_start + self.node_idx_remap,
                                          self.nodes_end + self.node_idx_remap])
                                          #- self.first_node  # subtract to account for 1st node index
                                          ).type(long)
        else:  # w/ reversed duplicate edges for undirected [graph] like CITE
            self.edge_idx = tensor(array([self.nodes_start + self.nodes_end + self.node_idx_remap,
                                          self.nodes_end + self.nodes_start + self.node_idx_remap])
                                          #- self.first_node  # subtract to account for 1st node index
                                          ).type(long)
        if set_node_labels:
            # assumes label values start at 0, unused for eval subgraph:
            #self.node_labels = tensor(array([one_hot_encoder(node[1]['node_label'], n_node_labels)  # one-hot encoded
            #                                 for node in self.nodes])).type(float)
            self.node_labels = tensor(array([node[1]['node_label']  # *not* one-hot encoded
                                             for node in self.nodes])).type(long) #.type(long) #.type(float)
        self.node_attributes = tensor(array([node[1]['node_attributes']
                                             for node in self.nodes])).type(float)  # *not* one-hot encoded
        if set_edge_labels:
            self.edge_labels = tensor(array([edge[2]['edge_label'] for edge in self.edges])).type(long) #.type(long) #.type(float)

    """# decomment to use in DataLoader, possibly needs to be adapted for LINK:
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        '''returns part of sparse representation of single, given [graph] all in one as tuple of th.tensors, index is redundant here'''
        #return super().__getitem__(index)

        # draw edge_idx & degree_factors from constructor:
        return self.edge_idx, self.degree_factors


    def __len__(self) -> int:
        '''length of dataset/batch, here just 1 for single [graph]'''
        return 1
    """



if __name__ == "__main__":
    # test sparse graph representation:
    import pickle

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    thresh = 5
    G = Sparse_Graph(graph, False)
    print(f"{G.n_nodes}\n{G.degree_factors[: thresh]}\n{G.edge_idx[: thresh]}")
    #\n{G.node_labels[: cutoff]}  #\n{G.edge_labels[: cutoff]}  #\n{G.nodes[: cutoff]}\n{G.edges[: cutoff]}\n{G.first_node}
