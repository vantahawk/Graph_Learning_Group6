
import torch as th
from torch.nn import Linear, Module, ModuleList, DataParallel
import torch.nn.functional as F
from torch.optim import Adam#, RMSprop
from torch.utils.data import DataLoader#, TensorDataset
from torch_scatter import scatter_sum, scatter_mean, scatter_max

# internal imports
from src.dataset import Custom_Dataset
from src.collation import custom_collate
from src.layer import GNN_Layer, activation_function
from src.pooling import Sum_Pooling
from src.virtual_node import Virtual_Node

class GNN(Module):
    '''module for the overall GNN, including series of [n_GNN_layers] GNN layers, each w/ optional virtual node, followed by a sum pooling layer & finally a graph-lvl MLP w/ [n_MLP_layers] layers; input dim. is composed of node & edge feature dim., output dim. is 1, matching the scalar, real-valued graph labels'''
    def __init__(self, scatter_type: str, use_virtual_nodes: bool,
                 n_MLP_layers: int, dim_MLP: int, n_virtual_layers: int,
                 n_GNN_layers: int, dim_between: int, dim_M: int, dim_U: int, n_M_layers: int, n_U_layers: int,
                 dim_node: int = 21, dim_edge: int = 3, dim_graph: int = 1, mlp_nonlin:str="relu", m_nonlin:str="relu", u_nonlin:str="relu", skip:bool=True,residual:bool=True, dropbout_prob:float=0.0) -> None:
        super().__init__()

        self.n_GNN_hidden = n_GNN_layers - 1  # number of hidden GNN layers
        self.n_MLP_hidden = n_MLP_layers - 1  # number of hidden MLP layers
        #self.use_virtual_nodes = use_virtual_nodes  # whether to use (True) or bypass (False) all virtual nodes

        #dropout-layer
        self.dropout = th.nn.Dropout(p=dropbout_prob)

        self.GNN_input = GNN_Layer(scatter_type, dim_node, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers, m_nonlin=m_nonlin, u_nonlin=u_nonlin, residual=residual)  # input GNN layer
        """
        # list of hidden GNN layers, i.e. GNN layers after input GNN layer
        self.GNN_hidden = ModuleList([GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)
                                      for layer in range(self.n_GNN_hidden)])

        if use_virtual_nodes:  # optional list of virtual nodes, one before each hidden GNN layer
            self.virtual_nodes = ModuleList([Virtual_Node(dim_between, n_virtual_layers) for layer in range(self.n_GNN_hidden)])
        """
        # use skip connection
        self.skip = skip

        # list of hidden GNN layers, i.e. GNN layers after input GNN layer, optionally each w/ prior virtual node
        if use_virtual_nodes:
            GNN_hidden = []
            for layer in range(self.n_GNN_hidden):
                GNN_hidden += [Virtual_Node(dim_between, n_virtual_layers),
                               GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers, m_nonlin=m_nonlin, u_nonlin=u_nonlin, residual=residual)]
            self.n_GNN_hidden *= 2  # double to account for added virtual nodes in forward fct.
        else:  # disable/bypass virtual nodes
            GNN_hidden = [GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers, m_nonlin=m_nonlin, u_nonlin=u_nonlin, residual=residual)
                          for layer in range(self.n_GNN_hidden)]
        self.GNN_hidden = ModuleList(GNN_hidden)


        self.sum_pooling = Sum_Pooling()  # sparse sum pooling layer, node- to graph-lvl


        self.mlp_nonlin = mlp_nonlin  # activation fct. for MLP layers
        # linear, graph-lvl MLP layers
        if n_MLP_layers > 1:  # for >=2 MLP layers
            # list of hidden MLP layers including prior input MLP layer
            self.MLP_hidden = ModuleList(
                [Linear(dim_between, dim_MLP, bias=True)] + 
                [Linear(dim_MLP, dim_MLP, bias=True) for layer in range(n_MLP_layers - 2)]
            )
            self.MLP_output = Linear(dim_MLP, dim_graph, bias=True)  # output MLP layer
        else:  # n_MLP_layers <= 1
            self.MLP_hidden = ModuleList([])  # no hidden MLP layers
            self.MLP_output = Linear(dim_between, dim_graph, bias=True)  # singular output MLP layer


    def forward(self, node_features: th.Tensor, edge_features: th.Tensor, edge_idx: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
        '''forward fct. of overall GNN, takes in node & edge features as well as edge index lists & batch_idx of graphs in given batch/dataset, returns predicted graph labels thereof'''
        y:th.Tensor = self.GNN_input(node_features, edge_features, edge_idx, batch_idx)  # apply input GNN layer
        """
        if self.use_virtual_nodes:
            for layer in range(self.n_GNN_hidden):
                y = self.virtual_nodes[layer](y, batch_idx)  # apply virtual nodes
                y = self.GNN_hidden[layer](y, edge_features, edge_idx)  # apply hidden GNN layers
        else:  # disable/bypass virtual nodes
            for layer in range(self.n_GNN_hidden):
                y = self.GNN_hidden[layer](y, edge_features, edge_idx)  # only apply hidden GNN layers
    	"""
        skip_val = y.clone()

        # apply hidden GNN layers w/ optional virtual nodes (version w/o if-statement in forward fct.)
        for layer in range(self.n_GNN_hidden):
            y = self.dropout(y)
            y = self.GNN_hidden[layer](y, edge_features, edge_idx, batch_idx)
            if self.skip:
                y += skip_val

        y = self.sum_pooling(y, batch_idx)  # apply sum pooling layer

        for layer in range(self.n_MLP_hidden):
            y = self.MLP_hidden[layer](y)  # apply hidden MLP layers
            y = activation_function[self.mlp_nonlin](y) # TODO try different activation fct.s...

        return self.MLP_output(y)  # apply linear output MLP layer

