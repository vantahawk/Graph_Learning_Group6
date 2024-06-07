import networkx as nx
import numpy as np
import torch as th
from torch.utils.data import Dataset

import random


def one_hot_encoder(label: int, length: int) -> np.ndarray:
    '''returns one-hot vector according to given label integer and length'''
    return np.eye(length)[label]



class Custom_Dataset(Dataset):
    '''implements custom [Dataset] class for given dataset (list of nx.graphs)'''
    def __init__(self, graphs: list[nx.Graph], size:int|None = None, seed:int=0) -> None:
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

        self.node_features = [th.tensor(np.array([one_hot_encoder(node[1]['node_label'], 21)
                                                  for node in graph.nodes(data=True)]), dtype=th.float)
                              for graph in graphs]  # one-hot encoded
        #self.node_features = [th.tensor(np.array([[node[1]['node_label']]
        #                                          for node in graph.nodes(data=True)])).type(th.float)
        #                      for graph in graphs]  # *not* one-hot encoded

        self.edge_features = [ th.tensor(
            [one_hot_encoder(edge[2]['edge_label'] - 1, 3) for _ in range(2)
                for edge in graph.edges(data=True) 
            ]) for graph in graphs
        ]  # one-hot encoded

        self.graph_labels = [th.tensor(np.array(graph.graph['label'][0]), dtype=th.float) for graph in graphs]  # scalar, real-valued


    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        '''returns custom sparse representation for each graph (index) as tuple of th.tensors'''
        #return super().__getitem__(index)

        # draw sparse representation from constructor for each graph index
        return self.edge_idx[index], self.node_features[index], self.edge_features[index], self.graph_labels[index]


    def __len__(self) -> int:
        '''length of dataset/batch, i.e. number of graphs in it'''
        return len(self.graphs)
