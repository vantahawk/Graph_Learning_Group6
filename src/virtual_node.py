import torch as th
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch_scatter import scatter_sum



class Virtual_Node(Module):
    '''module for single virtual node'''
    def __init__(self, dim: int, n_virtual_layers: int) -> None:
        super().__init__()

        self.n_virtual_layers = n_virtual_layers  # number of layers in MLP of virtual node
        self.virtual_MLP = ModuleList([Linear(dim, dim, bias=True) for layer in range(n_virtual_layers)])  # construct MLP of virtual node


    def forward(self, x: th.Tensor, edge_features: th.Tensor, edge_idx: th.Tensor, batch_idx: th.Tensor):
        '''forward fct. for single virtual node as described in exercise sheet/lecture script'''
        y = scatter_sum(x, batch_idx, dim=0)  # pool nodes in batch graph to their original graph w/ batch_idx

        for layer in range(self.n_virtual_layers):  # apply MLP of virtual node
            y = self.virtual_MLP[layer](y)
            y = F.relu(y)

        return x + y[batch_idx]  # expand graph-lvl embedding back to node-lvl w/ batch_idx & add it to original input
