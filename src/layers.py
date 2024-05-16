import torch as th
#import torch.nn.functional as F
#from preprocessing import max_node_dim#, norm_adjacency, zero_pad, stack_adjacency #  *
#import networkx as nx
#import torch.nn.functional as F
#import argparse


class GCN_Layer(th.nn.Module):
    '''module for single, linear, node-level GCN layer'''
    def __init__(self, dim_in, dim_out):
        super(GCN_Layer, self).__init__()

        # linear weight matrix initialized with kaiming init
        self.W = th.nn.Parameter(th.empty(dim_in, dim_out))
        th.nn.init.kaiming_normal_(self.W)

        """
        # additive bias initialized with zeros
        if self.add_bias:
            self.b = th.nn.Parameter(th.empty(<max_node_dim>, dim_out))
            th.nn.init.zeros_(self.b)
        """

    # expects input tensor x of shape (batch_size, dim_in)
    def forward(self, x, A):
        '''forward fct. for single GCN layer as described in exercise sheet/lecture script, has single-graph form here but is applied elem.wise to batch in practise, takes in norm.ed, zero-padded (& stacked) adj.mat.s [A]'''
        x = x.type(self.W.dtype)  # quick fix

        y = th.matmul(x, self.W) #+ self.b
        #y = th.bmm(x, self.W) #+ self.b
        #y = th.matmul(A, th.matmul(x, self.W)) #+ self.b
        #y = th.bmm(z, th.bmm(x, self.W)) #+ self.b

        A = A.type(y.dtype)  # quick fix
        #y = th.matmul(A, y)
        y = th.bmm(A, y)
        return (y, A)



#if __name__ == "__main__":  # TODO Ex.2 demo
