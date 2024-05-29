# external imports
import torch as th
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList

# internal imports
from src.layers import GCN_layer



class GCN_graph(Module):
    '''module for graph-level GCN (Ex.3) as described in exercise sheet/lecture script: stack of 5 (relu-activated) GCN single layers, 1 sum pooling layer & 2 graph-level MLP/linear layers, linear output'''
    def __init__(self, input_dim: int, output_dim: int, dropout_list: list[int], hidden_dim: int = 64, n_GCN_layers: int = 5) -> None:
        super().__init__()

        self.n_hidden_GCN_layers = n_GCN_layers - 1  # number of hidden GCN layers

        # 5 GCN layers w/ 4 form-identical hidden layers
        self.input_layer = GCN_layer(input_dim, hidden_dim, dropout_list[0])  # input layer
        self.hidden_layers = ModuleList(
            [GCN_layer(hidden_dim, hidden_dim, dropout_list[layer]) for layer in range(1, self.n_hidden_GCN_layers + 1)])

        # graph-level MLP = hidden layer & linear output
        self.hidden_MLP_layer = Linear(hidden_dim, hidden_dim, bias=True)
        self.output_layer = Linear(hidden_dim, output_dim, bias=True)


    def forward(self, x: th.Tensor, A: th.Tensor) -> th.Tensor:
        '''forward fct. for whole graph-level GCN (Ex.3)'''
        y = F.relu(self.input_layer(x, A))

        for i in range(self.n_hidden_GCN_layers):
            y = F.relu(self.hidden_layers[i](y, A))

        y = th.sum(y, -2)  # sum pooling layer <=> summing along node-axis

        y = self.hidden_MLP_layer(y)
        y = F.relu(y)  # optional...?
        return self.output_layer(y)  # graph-level output w/o activation



class GCN_node(Module):
    '''module for node-level GCN (Ex.4) as described in exercise sheet/lecture script: stack of 3 (relu-activated) GCN single layers & 1 node-level output layer, linear output'''
    def __init__(self, input_dim: int, output_dim: int, dropout_list: list[int], hidden_dim: int = 64, n_GCN_layers: int = 3) -> None:
        super().__init__()

        self.n_hidden_GCN_layers = n_GCN_layers - 1  # number of hidden GCN layers

        # 3 GCN layers w/ 2 form-identical hidden layers
        self.input_layer = GCN_layer(input_dim, hidden_dim, dropout_list[0])  # input layer
        self.hidden_layers = ModuleList(
            [GCN_layer(hidden_dim, hidden_dim, dropout_list[layer]) for layer in range(1, self.n_hidden_GCN_layers + 1)])

        # node-level MLP = 1 linear output layer
        self.output_layer = Linear(hidden_dim, output_dim, bias=True)


    def forward(self, x: th.Tensor, A: th.Tensor) -> th.Tensor:
        '''forward fct. for whole graph-level GCN (Ex.3)'''
        y = F.relu(self.input_layer(x, A))

        for i in range(self.n_hidden_GCN_layers):
            y = F.relu(self.hidden_layers[i](y, A))

        return self.output_layer(y)  # node-level output w/o activation
