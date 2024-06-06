import numpy as np
import torch as th
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_max



def scatter_max_shortcut(src: th.Tensor, idx: th.Tensor, dim: int):
    '''shortcut for scatter_max to isolate max. values from arguments, for avoiding if-statement in forward-fct. of GNN_layer'''
    return scatter_max(src, idx, dim=dim)[0]



def soft_relu(x: th.Tensor) -> th.Tensor:
    '''custom activation fct. as simple, non-linear, cont. diff.able alternative to ReLU'''
    return th.mul(F.relu(x), F.softsign(x))  # multiplies elem.wise ReLU & Softsign, formula: [x > 0] * x² / (1 + x)



def skewed_identity(x: th.Tensor) -> th.Tensor:
    '''custom activation fct. as simple, non-linear, cont. diff.able, anti-sym. alternative to identity, similar to tanhshrink'''
    return x - F.softsign(x)  # subtracts elem.wise Softsign from x, formula: x - x / (1 + |x|)
    #return x + F.softsign(x)  # adds elem.wise Softsign to x, formula: x + x / (1 + |x|)



def simple_elu(x: th.Tensor) -> th.Tensor:
    '''custom activation fct. as simple, non-linear, twice cont. diff.able alternative to ReLU & ELU, similar to Softplus (shifted down)'''
    return th.mul(x, F.softsign(x) + 1) / 2  # all op.s elem.wise, formula: x * (1 + x / (1 + |x|)) / 2
    #return th.mul(x, F.softsign(x) + 1)  # scaled, formula: x * (1 + x / (1 + |x|))
    #return th.mul(x, F.softsign(x) + 1) / 2 + 0.5  # shifted up, formula: (1 + x * (1 + x / (1 + |x|))) / 2



class GNN_Layer(Module):
    '''module for single, node-level GNN layer'''
    def __init__(self, scatter_type: str, activation_M: str, dim_in: int, dim_edge: int, dim_M: int, dim_U: int, dim_out: int, n_M_layers: int, n_U_layers: int) -> None:
        super().__init__()

        self.scatter_type = scatter_type

        self.n_M_hidden = n_M_layers - 1  # number of hidden single, linear layers in M
        self.n_U_hidden = n_U_layers - 1  # number of hidden single, linear layers in U

        # activation fct. for each layer of M
        if activation_M == 'softsign':
            self.activation_M = F.softsign  # simple, sigmoid-like: x / (1 + |x|), okay?
        elif activation_M == 'softplus':
            self.activation_M = F.softplus  # smooth alternative to ReLU, default: beta=1, threshold=20, fairly promising?
        elif activation_M == 'elu':
            self.activation_M = F.elu  # similar to Softplus & ReLU (-1 -> origin -> identity), default: alpha=1, okay?
        elif activation_M == 'tanh':
            self.activation_M = F.tanh  # sigmoid-like, alternative to softsign, fairly promising?
        elif activation_M == 'tanhshrink':
            self.activation_M = F.tanhshrink  # x - tanh, promising?
        # own creations...
        elif activation_M == 'soft_relu':
            self.activation_M = soft_relu  # less promising?
        elif activation_M == 'skewed_identity':
            self.activation_M = skewed_identity  # promising?
        elif activation_M == 'simple_elu':
            self.activation_M = simple_elu  # promising?
        # default: relu
        else:
            self.activation_M = F.relu  # promising?

        # activation fct. for each layer of U
        self.activation_U = F.relu

        # single, linear, node-level layers for MLPs M (message)...
        self.M_input = Linear(dim_in + dim_edge, dim_M, bias=True)
        self.M_hidden = ModuleList([Linear(dim_M, dim_M, bias=True) for layer in range(self.n_M_hidden)])
        # ...& U (update)
        if n_U_layers > 1:  # for >=2 U-layers
            # list of hidden U-layers including prior input layer of U
            self.U_hidden = ModuleList(
                [Linear(dim_in + dim_M, dim_U, bias=True)] + [Linear(dim_U, dim_U, bias=True) for layer in range(n_U_layers - 2)])
            self.U_output = Linear(dim_U, dim_out, bias=True)  # output layer of U
        else:  # n_U_layers <= 1
            self.U_hidden = ModuleList([])  # no hidden U-layers
            self.U_output = Linear(dim_in + dim_M, dim_out, bias=True)  # singular output layer of U

        # choose scatter aggregation type for message passing
        if scatter_type == 'sum':
            self.scatter = scatter_sum
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:  # elif scatter_type == 'max'
            self.scatter = scatter_max_shortcut


    def forward(self, x: th.Tensor, edge_features: th.Tensor, edge_idx: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
        '''forward fct. for single GNN layer as described in exercise sheet/lecture script, uses scatter-operations w/ edge_idx as well as edge_features for message-passing'''
        # select node-level input by start nodes (edge_idx[0]) & concatenate them w/ edge_features
        y = th.cat([x[edge_idx[0]], edge_features], -1)#.type(th.float)

        y = self.M_input(y)  # apply input layer of M
        y = self.activation_M(y)

        for layer in range(self.n_M_hidden):  # apply hidden layers of M
            y = self.M_hidden[layer](y)
            y = self.activation_M(y)

        # select node-level input by end nodes (edge_idx[1]), scatter output from M according to edge_idx & concatenate both
        z = th.cat([x, self.scatter(y, edge_idx[1], dim=0)], -1)

        for layer in range(self.n_U_hidden):  # apply hidden layers of U
            z = self.U_hidden[layer](z)
            z = self.activation_U(z)

        return self.U_output(z)  # apply linear output layer of U
