'''The entry point of our implementation'''

# external imports
import argparse
import networkx as nx
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error
#from sympy import Line
from sympy import use
import torch as th
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_scatter import scatter_sum, scatter_mean, scatter_max

# internal imports
from src.dataset import Custom_Dataset
from src.collation import custom_collate
from src.layer import GNN_Layer
from src.pooling import Sum_Pooling
from src.virtual_node import Virtual_Node



class GNN(Module):
    '''module for the overall GNN, including series of [n_GNN_layers] GNN layers, each w/ optional virtual node, followed by a sum pooling layer & finally a graph-lvl MLP w/ [n_MLP_layers] layers; input dim. is composed of node & edge feature dim., output dim. is 1, matching the scalar, real-valued graph labels'''
    def __init__(self, scatter_type: str, use_virtual_nodes: bool,
                 n_MLP_layers: int, dim_MLP: int, n_virtual_layers: int,
                 n_GNN_layers: int, dim_between: int, dim_M: int, dim_U: int, n_M_layers: int, n_U_layers: int,
                 dim_node: int = 21, dim_edge: int = 3, dim_graph: int = 1) -> None:
        super().__init__()

        self.n_GNN_hidden = n_GNN_layers - 1  # number of hidden GNN layers
        self.n_MLP_hidden = n_MLP_layers - 1  # number of hidden MLP layers
        #self.use_virtual_nodes = use_virtual_nodes  # whether to use (True) or bypass (False) all virtual nodes

        self.GNN_input = GNN_Layer(scatter_type, dim_node, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)  # input GNN layer
        """
        # list of hidden GNN layers, i.e. GNN layers after input GNN layer
        self.GNN_hidden = ModuleList([GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)
                                      for layer in range(self.n_GNN_hidden)])

        if use_virtual_nodes:  # optional list of virtual nodes, one before each hidden GNN layer
            self.virtual_nodes = ModuleList([Virtual_Node(dim_between, n_virtual_layers) for layer in range(self.n_GNN_hidden)])
        """

        # list of hidden GNN layers, i.e. GNN layers after input GNN layer, optionally each w/ prior virtual node
        if use_virtual_nodes:
            GNN_hidden = []
            for layer in range(self.n_GNN_hidden):
                GNN_hidden += [Virtual_Node(dim_between, n_virtual_layers),
                               GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)]
            self.n_GNN_hidden *= 2  # double to account for added virtual nodes in forward fct.
        else:  # disable/bypass virtual nodes
            GNN_hidden = [GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)
                          for layer in range(self.n_GNN_hidden)]
        self.GNN_hidden = ModuleList(GNN_hidden)

        self.sum_pooling = Sum_Pooling()  # sparse sum pooling layer, node- to graph-lvl

        # linear, graph-lvl MLP layers
        if n_M_layers > 1:  # for >=2 MLP layers
            # list of hidden MLP layers including prior input MLP layer
            self.MLP_hidden = ModuleList(
                [Linear(dim_between, dim_MLP, bias=True)] + [Linear(dim_MLP, dim_MLP, bias=True) for layer in range(n_MLP_layers - 2)])
            self.MLP_output = Linear(dim_MLP, dim_graph, bias=True)  # output MLP layer
        else:  # n_MLP_layers <= 1
            self.MLP_hidden = ModuleList([])  # no hidden MLP layers
            self.MLP_output = Linear(dim_between, dim_graph, bias=True)  # singular output MLP layer


    def forward(self, node_features: th.Tensor, edge_features: th.Tensor, edge_idx: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
        '''forward fct. of overall GNN, takes in node & edge features as well as edge index lists & batch_idx of graphs in given batch/dataset, returns predicted graph labels thereof'''
        y = self.GNN_input(node_features, edge_features, edge_idx)  # apply input GNN layer
        """
        if self.use_virtual_nodes:
            for layer in range(self.n_GNN_hidden):
                y = self.virtual_nodes[layer](y, batch_idx)  # apply virtual nodes
                y = self.GNN_hidden[layer](y, edge_features, edge_idx)  # apply hidden GNN layers
        else:  # disable/bypass virtual nodes
            for layer in range(self.n_GNN_hidden):
                y = self.GNN_hidden[layer](y, edge_features, edge_idx)  # only apply hidden GNN layers
    	"""

        # apply hidden GNN layers w/ optional virtual nodes (version w/o if-statement in forward fct.)
        for layer in range(self.n_GNN_hidden):
            y = self.GNN_hidden[layer](y, edge_features, edge_idx, batch_idx)

        y = self.sum_pooling(y, batch_idx)  # apply sum pooling layer

        for layer in range(self.n_MLP_hidden):
            y = self.MLP_hidden[layer](y)  # apply hidden MLP layers
            y = F.relu(y)  # TODO optional?, different activation fct.?

        return self.MLP_output(y)  # apply linear output MLP layer



#def main() -> None:
    # TODO finish evaluation...



#if __name__ == "__main__":  # TODO command line arguments
