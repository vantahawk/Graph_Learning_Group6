'''GNN layer adapted for node-feature-only graphs (like CITE)'''
# external imports:
from torch import dtype, Tensor, float, float16, float32, float64, tensor
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_max

# internal imports:
#from sparse_graph import Sparse_Graph



def scatter_max_shortcut(src: Tensor, idx: Tensor, dim: int):
    '''shortcut for scatter_max to isolate max. values from arguments, for avoiding if-statement in forward-fct. of GNN_layer'''
    return scatter_max(src, idx, dim=dim)[0]



class GNN_Layer(Module):
    '''module for single, node-level GNN layer w/ M simplified to identity/linear projection, similar to GCN for scatter_type = scatter_sum/mean'''
    def __init__(self, #G: Sparse_Graph,
                 dim_in: int, dim_U: int, dim_out: int, n_U_layers: int,  # standard param.s for U
                 n_pass: int = 1,  # number of message passes per GNN layer, <=> powers of (norm.ed/stochastic) adj.mat. for scatter_sum/mean
                 scatter_type: str = 'sum',  # scatter aggregation type for message passing
                 dtype: dtype = float64 #float16 #float32  #float64  # data type for th.nn.Linear-layers
                 ) -> None:
        super().__init__()

        # key attributes:
        self.n_pass = n_pass
        self.n_U_hidden = n_U_layers - 1  # number of hidden single, linear layers in U
        self.activation_U = F.relu  # default activation fct. for all U-layers

        # single, linear, node-level layers for MLPs U (update):
        if n_U_layers > 1:  # for >=2 U-layers
            # list of hidden U-layers including prior input layer of U:
            self.U_hidden = ModuleList(
                [Linear(dim_in, dim_U, bias=True, dtype=dtype)] + [Linear(dim_U, dim_U, bias=True, dtype=dtype)
                                                                   for layer in range(n_U_layers - 2)])
            self.U_output = Linear(dim_U, dim_out, bias=True, dtype=dtype)  # output layer of U
        else:  # n_U_layers <= 1
            self.U_hidden = ModuleList([])  # no hidden U-layers
            self.U_output = Linear(dim_in, dim_out, bias=True, dtype=dtype)  # singular output layer of U

        # choose scatter aggregation type for message passing:
        if scatter_type == 'sum':
            self.scatter = scatter_sum
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:  # elif scatter_type == 'max'
            self.scatter = scatter_max_shortcut


    def forward(self, y: Tensor,  # node_attributes-derived input
                #G: Sparse_Graph
                #, edge_idx: Tensor, degree_factors: Tensor  # decomment to use w/ DataLoader & .to(device) instead
                start_nodes: Tensor, end_nodes: Tensor, degree_factors_start: Tensor, degree_factors: Tensor  # for message passing on separate device
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  #tuple[Tensor, Sparse_Graph]:
        '''forward fct. for single GNN layer, uses scatter-operations w/ start_nodes & end_nodes from edge_idx for message-passing'''
        """#
        start_nodes, end_nodes = G.edge_idx[0], G.edge_idx[1]
        degree_factors_start = G.degree_factors_start
        degree_factors_end = G.degree_factors
        """#
        for step in range(self.n_pass):  # scatter node attributes via [n_pass] message passes

            # select node-level input by start nodes (edge_idx[0]):
            #y = y[start_nodes]  ## simple message passing <=> M = identity
            ## message passing w/ degree normalization like in GCN, originally meant for scatter_sum:
            y = y[start_nodes] * degree_factors_start

            ## scatter start-node-selected (& norm.ed) input (output from linear M) towards end nodes (edge_idx[1]):
            #y = self.scatter(y, end_nodes, dim=0)  ## simple message passing, scatter_mean <=> row/column-stochastic adj.mat.
            y = self.scatter(y, end_nodes, dim=0) * degree_factors  # message passing w/ degree normalization like in GCN, 2nd part

            #y = y.type(float32)  #float # re-cast (error-fix)

        for layer in range(self.n_U_hidden):  # apply hidden layers of U
            y = self.U_hidden[layer](y)
            y = self.activation_U(y)

        #return self.U_output(y), G
        return self.U_output(y), start_nodes, end_nodes, degree_factors_start, degree_factors  # apply linear output layer of U, pass G along
