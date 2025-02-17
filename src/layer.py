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

activation_function = {
    "relu": F.relu,
    "soft_relu": soft_relu,
    "skewed_identity": skewed_identity,
    "simple_elu": simple_elu,
    "leaky_relu": F.leaky_relu,
    "tanh": F.tanh,
    "celu":F.celu,
    "selu":F.selu,
}

class GNN_Layer(Module):
    '''module for single, node-level GNN layer'''
    def __init__(self, scatter_type: str, dim_in: int, dim_edge: int, dim_M: int, dim_U: int, dim_out: int, n_M_layers: int, n_U_layers: int, m_nonlin:str="relu", u_nonlin:str="relu", residual:bool=True) -> None:
        """A single GNN layer with a lot of options:
        
        Args:
            scatter_type (str): Type of scatter operation to use. One of 'sum', 'mean', 'max'.
            dim_in (int): Dimension of input features.
            dim_edge (int): Dimension of edge features.
            dim_M (int): Dimension of message MLP.
            dim_U (int): Dimension of update MLP.
            dim_out (int): Dimension of output features.
            n_M_layers (int): Number of layers in message MLP.
            n_U_layers (int): Number of layers in update MLP.
            m_nonlin (str): Activation function for message MLP.
            u_nonlin (str): Activation function for update MLP.
            residual (bool): Whether to use a residual connection.
        """

        super().__init__()
        self.residual = residual
        self.dim_in = dim_in
        self.dim_M = dim_M
        self.dim_out = dim_out
        if residual and self.dim_in + self.dim_M != self.dim_out:
            self.projection = Linear(dim_in + dim_M, dim_out, bias=False)
        self.scatter_type = scatter_type

        self.n_M_hidden = n_M_layers - 1  # number of hidden single, linear layers in M
        self.n_U_hidden = n_U_layers - 1  # number of hidden single, linear layers in U

        # activation fct. for each layer of M
        self.activation_M = activation_function[m_nonlin]  

        # activation fct. for each layer of U
        self.activation_U = activation_function[u_nonlin] 

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

        self.batch_norm = th.nn.BatchNorm1d(dim_out)


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
        if self.residual:
            res = z # store residual connection

        for layer in range(self.n_U_hidden):  # apply hidden layers of U
            z = self.U_hidden[layer](z)
            z = self.activation_U(z)


        # apply linear output layer of U, but with potential residual connection
        if self.residual:
            if self.dim_in + self.dim_M != self.dim_out:
                residual =  self.U_output(z)+self.projection(res)
            else:
                residual = self.U_output(z)+res  
        else:
            residual = self.U_output(z)

        #apply batch norm
        residual = self.batch_norm(residual)

        return residual