#external imports
import torch as th
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
#import argparse

#internal imports
from src.layers import GCN_Layer

#TODO global: consider adding dropout layers for performance improvements...



class GCN_graph(th.nn.Module):
    '''module for graph-level GCN (Ex.3) as described in exercise sheet/lecture script: stack of 5 (relu-activated) GCN single layers, 1 sum pooling layer & 2 graph-level MLP/linear layers, linear output'''
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, n_GCN_layers: int = 5):
        super(GCN_graph, self).__init__()

        self.n_hidden_GCN_layers = n_GCN_layers - 1  # number of hidden GCN single layers

        # 5 GCN layers w/ 4 form-identical hidden layers
        self.input_layer = GCN_Layer(input_dim, hidden_dim)  # input layer
        self.hidden_GCN_layers = th.nn.ModuleList([GCN_Layer(hidden_dim, hidden_dim) for layer in range(self.n_hidden_GCN_layers)])

        # graph-level MLP
        self.hidden_MLP_layer = th.nn.Linear(hidden_dim, hidden_dim, True)  # bias=True
        self.output_layer = th.nn.Linear(hidden_dim, output_dim, True)  # bias=True


    def forward(self, x: th.Tensor, A: th.Tensor) -> th.Tensor:
        '''forward fct. for whole graph-level GCN (Ex.3)'''
        y = F.relu(self.input_layer(x, A)[0])
        for i in range(self.n_hidden_GCN_layers):
            y = F.relu(self.hidden_GCN_layers[i](y, A)[0])

        y = th.sum(y, -2)  # sum pooling layer <=> summing along node-axis

        y = self.hidden_MLP_layer(y)  # graph-level hidden layer
        y = F.relu(y)  # optional...?
        return self.output_layer(y)  # graph-level output w/o activation



class GCN_node(th.nn.Module):
    '''module for node-level GCN (Ex.4) as described in exercise sheet/lecture script: stack of 3 (relu-activated) GCN single layers & 1 node-level output layer, linear output'''
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, n_GCN_layers: int = 3):
        super(GCN_node, self).__init__()

        self.n_hidden_GCN_layers = n_GCN_layers - 1  # total number of hidden GCN single layers

        # 3 GCN layers w/ 2 form-identical hidden layers
        self.input_layer = GCN_Layer(input_dim, hidden_dim)  # input layer
        self.hidden_layers = th.nn.ModuleList(
            [GCN_Layer(hidden_dim, hidden_dim) for layer in range(self.n_hidden_GCN_layers)])

        # node-level MLP = 1 linear output layer
        self.output_layer = th.nn.Linear(hidden_dim, output_dim, True)  # bias=True


    def forward(self, x: th.Tensor, A: th.Tensor) -> th.Tensor:
        '''forward fct. for whole graph-level GCN (Ex.3)'''
        y = F.relu(self.input_layer(x, A)[0])
        for i in range(self.n_hidden_GCN_layers):
            y = F.relu(self.hidden_layers[i](y, A)[0])

        return self.output_layer(y)  # node-level output w/o activation



#if __name__ == "__main__":  # TODO Ex.3/4 demo
