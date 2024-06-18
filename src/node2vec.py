import networkx as nx
import numpy as np
import torch as th
from torch.nn import Module, Parameter
#import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, kaiming_uniform_



class Node2Vec(Module):
    '''node2vec embedding as torch module w/ embedding matrix X as parameter & the def. stochastic loss fct. as foward fct.'''
    def __init__(self, graph: nx.Graph, dim: int, l: int) -> None:
        super().__init__()

        # main attributes
        self.graph = graph
        self.n_nodes = nx.number_of_nodes(graph)
        self.dim = dim  # embedding dimension
        self.l = l  # random walk length

        # initialize node2vec embedding matrix randomly as parameter  # TODO look for & test other initializations
        self.X = Parameter(th.empty(self.n_nodes, dim))
        kaiming_normal_(self.X)
        #kaiming_uniform_(self.X)


    def forward(self, rw_vec: th.Tensor) -> th.Tensor:
        '''forward fct. of node2vec embedding, takes stacked pq-walk data vector (1D), returns scalar value of stochastic loss fct. over (formally) one pq-walk (simplified, see conversion) as def. in sheet/script'''
        X_start = self.X[rw_vec[0]]  # embedding vec. of start node (X_s)
        walk_idx = rw_vec[: self.l + 1]  # selection from X acc. to pq-walk nodes, including start node
        neg_idx = rw_vec[self.l + 1 :]  # selection from X acc. to negative samples
        numerator_term = th.sum(th.matmul(self.X[walk_idx[1 :]], X_start), -1)  # see conversion of loss-fct., using walk_idx w/o start node

        # FIXME whether to reassign walk_idx to only include each node once (i.e. interpret pq-walk "w" as set rather than sequence in denominator term), interpretation in script/sheet unclear, enable/disable accordingly
        walk_idx = th.tensor(list(set(np.array(walk_idx))))

        # compute denominator term (see conversion of loss-fct.), then subtract numerator term
        return self.l * th.log(th.sum(th.exp(th.matmul(self.X[th.cat([walk_idx, neg_idx], -1)], X_start)), -1)) - numerator_term

"""
    def forward(self, rw_vec: th.Tensor) -> th.Tensor:
        '''short version of forward fct. of node2vec embedding, interpreting pq-walk "w" as sequence rather than set within denominator term'''
        X_start = self.X[rw_vec[0]]
        return self.l * th.log(th.sum(th.exp(th.matmul(self.X[rw_vec], X_start)), -1)) - th.sum(th.matmul(self.X[rw_vec[1 : self.l + 1]], X_start), -1)
"""
