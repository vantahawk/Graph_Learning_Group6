import torch as th
#import torch.nn.functional as F
from torch.nn import Dropout
#from preprocessing import max_node_dim#, norm_adjacency, zero_pad, stack_adjacency #  *
#import networkx as nx
#import torch.nn.functional as F
#import argparse



class GCN_Layer(th.nn.Module):
    '''module for single, linear, node-level GCN layer, here: optionally preceded by dropout layer'''
    def __init__(self, dim_in: int, dim_out: int, dropout_index: int):
        super(GCN_Layer, self).__init__()

        # linear weight matrix initialized with kaiming init
        self.W = th.nn.Parameter(th.empty(dim_in, dim_out))
        th.nn.init.kaiming_normal_(self.W)

        # dropout layer
        #self.dropout_layer = Dropout(p=0.5)  # TODO try different dropout probab. values p
        #self.dropout_index = dropout_index


    # expects input tensor x of shape (batch_size, dim_in) and prepared adj.mat. A of shape (batch_size, max_n_nodes, max_n_nodes)
    def forward(self, x: th.Tensor, A: th.Tensor) -> th.Tensor:
        '''forward fct. for single GCN layer as described in exercise sheet/lecture script, has single-graph form here but is applied elem.wise to batch in practise, takes in norm.ed, zero-padded (& stacked) adj.mat.s [A], here: optionally preceded by dropout layer'''

        #if self.dropout_index == 1:  # feed through dropout layer first if indexed
        #    x = self.dropout_layer(x)

        x = x.type(self.W.dtype)  # quick fix
        y = th.matmul(x, self.W)  # apply weight matrix multiplication

        A = A.type(y.dtype)  # quick fix
        y = th.bmm(A, y)  # apply message passing w/ adj.mat. multiplication

        return y



#if __name__ == "__main__":  # optional Ex.2 demo
