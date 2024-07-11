'''implememtation for sparse representaion of graph (origially intended for CITE)'''
from networkx import DiGraph, MultiDiGraph, Graph, MultiGraph, adjacency_matrix, is_directed, relabel_nodes
import numpy as np
from numpy import array, sum
from torch import float64, long, tensor



class Sparse_Graph():  # generic class
    '''implements sparse representation of given, *single* nx.[graph], originally for Graph (like CITE), partially adapted for DiGraph & MulitDiGraph (like LINK) as well, would still need to deal w/ "id" of edges in LINK'''
    def __init__(self, graph: Graph | MultiGraph | DiGraph | MultiDiGraph, set_node_labels: bool, set_edge_labels: bool = False
                 #, n_node_labels: int = 4  # 4 node label classes in CITE
                 ) -> None:

        self.n_nodes = graph.number_of_nodes()
        self.node_idx = [node[0] for node in graph.nodes(data=True)]  # list of node indices
        # node index remapping:
        self.node_map = {self.node_idx[i] : i for i in range(self.n_nodes)}  # node remapping dictionary w/ elem.s of form (old_idx: new_idx)
        graph = relabel_nodes(graph, self.node_map, copy=True)  # equiv. [graph] but w/ remapped node indices
        #
        # key attributes:
        self.set_node_labels = set_node_labels
        self.nodes = graph.nodes(data=True)
        self.edges = graph.edges(data=True)
        #self.first_node = min(self.node_idx)  # account for node count via 1st node index
        self.node_idx_remap = list(range(self.n_nodes))  # remapped node indices = [0,...,n_nodes-1] (order not important here)

        # pre-compute sparse representations of given [graph]:
        self.adj_mat = adjacency_matrix(graph)#.toarray()
        # for degree normalization like in GCN but for using scatter:
        self.degree_factors = tensor((sum(self.adj_mat.toarray(), axis=-1) + 1) ** -0.5).reshape((self.n_nodes, 1)).type(float64)
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
                                          self.nodes_end + self.nodes_start + self.node_idx_remap])#.astype(int32)
                                          #- self.first_node  # subtract to account for 1st node index
                                          ).type(long)
        self.degree_factors_start = self.degree_factors[self.edge_idx[0]]  # slice of degree_factors w.r.t. start nodes in edge_idx
        if set_node_labels:
            self.node_labels = tensor(array([node[1]['node_label']  # *not* one-hot encoded
                                             for node in self.nodes])).type(long) #.type(long) #.type(float)
        self.node_attributes = tensor(array([node[1]['node_attributes']
                                             for node in self.nodes])).type(float64) #.type(float)  # *not* one-hot encoded
        if set_edge_labels:
            self.edge_labels = tensor(array([edge[2]['edge_label'] for edge in self.edges])).type(long) #.type(long) #.type(float)



if __name__ == "__main__":
    # test sparse graph representation:
    import pickle
    from timeit import default_timer

    with open('datasets/CITE/data.pkl', 'rb') as data:
        graph = pickle.load(data)

    thresh = 5
    t_start = default_timer()
    G = Sparse_Graph(graph, False)
    print(f"Time = {default_timer() - t_start} secs\n{G.n_nodes}\n{G.degree_factors[: thresh]}\n{G.edge_idx[: thresh]}")
