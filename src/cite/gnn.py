'''GNN module adapted for single node-feature-only graphs (like CITE)'''
# external imports:
#import argparse
#import networkx as nx
#from networkx import Graph
#import numpy as np
from numpy import argmax#, empty
#from psutil import cpu_count
#from sklearn.metrics import accuracy_score#, mean_absolute_error
from pandas import cut
import torch as th  # TODO replace th. w/ direct imports (maybe)
from torch import Tensor, cat
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch.optim import Adam#, RMSprop
#from torch.utils.data import DataLoader, TensorDataset
#from torch_scatter import scatter_sum, scatter_mean, scatter_max

# internal imports:
from sparse_graph import Sparse_Graph
#from collation import custom_collate
from layer import GNN_Layer
#from pooling import Sum_Pooling
#from virtual_node import Virtual_Node



class GNN(Module):
    '''module for the overall GNN; passes thru series of [n_GNN_layers] GNN layers w/ [n_pass] message passes per layer, then concatenates w/ node2vec-embedding & finally passes thru node-lvl MLP w/ [n_MLP_layers] layers; input = [node_attributes], output = one-hot encoded [node_labels]; adapted for CITE (single, undirected graph); alternatively embedding concatenation can also be placed b4 GNN layers de/re-commenting lines marked by "###" accordingly'''
    def __init__(self, #G: Sparse_Graph,  # sparse graph rep. for node_attributes & edge_idx
                 n_MLP_layers: int, dim_MLP: int,  # post-MLP
                 dim_n2v: int,  # dimension of node2vec-embedding
                 n_GNN_layers: int, dim_between: int, dim_U: int, n_U_layers: int,  # standard param.s for GNN layers
                 l: int = 8, # max. length of closed walks in CW-embedding
                 n_pass: int = 1,  # number of message passes per GNN layer, <=> powers of (norm.ed/stochastic) adj.mat. for scatter_sum/mean
                 scatter_type: str = 'sum',  # scatter aggregation type for message passing
                 dtype: th.dtype = th.float64, #th.float16 #th.float32 #th.float64  # data type for th.nn.Linear-layers
                 dim_attr: int = 1639, dim_out: int = 4  # node_attributes-dim & number of label classes resp.
                 ) -> None:
        super().__init__()

        dim_cw = l - 1  # CW-embedding dimension

        self.n_GNN_hidden = n_GNN_layers - 1  # number of hidden GNN layers
        self.n_MLP_hidden = n_MLP_layers - 1  # number of hidden MLP layers
        #self.use_virtual_nodes = use_virtual_nodes  # whether to use (True) or bypass (False) all virtual nodes
        self.activation_MLP = F.relu  # activation fct. for each hidden layer of MLP

        self.GNN_input = GNN_Layer(dim_attr, dim_U, dim_between, n_U_layers, n_pass, scatter_type)  ### for embedding-concat. *after* GNN layers
        ### for embedding-concat. *before* GNN layers:
        #self.GNN_input = GNN_Layer(dim_attr + dim_n2v + dim_cw, dim_U, dim_between, n_U_layers, n_pass, scatter_type)

        # list of hidden GNN layers, i.e. GNN layers after input GNN layer:
        self.GNN_hidden = ModuleList([GNN_Layer(dim_between, dim_U, dim_between, n_U_layers, n_pass, scatter_type)
                                      for layer in range(self.n_GNN_hidden)])

        # linear, graph-lvl MLP layers:
        if n_MLP_layers > 1:  # for >=2 MLP layers
            # list of hidden MLP layers including prior input MLP layer:
            self.MLP_hidden = ModuleList(
                [Linear(dim_between + dim_n2v + dim_cw, dim_MLP, bias=True, dtype=dtype)]  ### for embedding-concat. *after* GNN layers
                #[Linear(dim_between, dim_MLP, bias=True, dtype=dtype)]  ### for embedding-concat. *before* GNN layers
                + [Linear(dim_MLP, dim_MLP, bias=True, dtype=dtype) for layer in range(n_MLP_layers - 2)])
            self.MLP_output = Linear(dim_MLP, dim_out, bias=True, dtype=dtype)  # output MLP layer
        else:  # n_MLP_layers <= 1
            self.MLP_hidden = ModuleList([])  # no hidden MLP layers
            # singular output MLP layer:
            self.MLP_output = Linear(dim_between + dim_n2v + dim_cw, dim_out, bias=True, dtype=dtype)  ### for emb.-concat. *after* GNN layers
            #self.MLP_output = Linear(dim_between, dim_out, bias=True, dtype=dtype)  ### for emb.-concat. *before* GNN layers


    def forward(self, G: Sparse_Graph,
                #, edge_idx: Tensor, degree_factors: Tensor,  # decomment to use w/ DataLoader & .to(device) instead, adapt fct. block accordingly
                X: Tensor,  # (slice of) node2vec-embedding tensor
                W: Tensor  # (slice of) CW-embedding tensor
                ) -> Tensor:
        '''forward fct. of overall GNN, takes in [node_attributes] & node2vec-embedding tensor, returns predicted node labels thereof'''
        # primary input:
        y = G.node_attributes  ### start w/ node_attributes only, concatenate node2vec- & CW-embedding *after* GNN layers
        #y = cat([G.node_attributes, X, W], -1)  ### concatenate node_attributes w/ node2vec- & CW-embedding *before* GNN layers

        y, G = self.GNN_input(y, G)  # apply input GNN layer to primary input, pass G along

        for layer in range(self.n_GNN_hidden):  # apply hidden GNN layers, pass G along
            y, G = self.GNN_hidden[layer](y, G)

        y = cat([y, X, W], -1)  ### secondary input *after* GNN layers: concatenate (primary) output w/ node2vec- & CW-embedding

        #y = y.type(th.float32)  #float # re-cast (error-fix)

        for layer in range(self.n_MLP_hidden):  # apply hidden MLP layers to secondary input
            y = self.MLP_hidden[layer](y)
            y = self.activation_MLP(y)

        return self.MLP_output(y)  # apply linear output MLP layer; return predicted, node_label logits as (secondary) output



def accuracy(z_pred: Tensor, z_true: Tensor) -> float:
    '''returns accuracy between predicted class label logits vs. true class label indices (in range [0,...,len(z_pred)-1])'''
    return (z_pred.argmax(-1) == z_true).type(th.float).mean().item()
    #return (F.softmax(z_pred, dim=-1).argmax(-1) == z_true).type(th.float).mean().item()



def run_model(G_train: Sparse_Graph, G_val: Sparse_Graph,  # sparse graph rep.s for training & validation
              X_train: Tensor, X_val: Tensor,  # node2vec embedding tensors for training & validation
              W_train: Tensor, W_val: Tensor,  # CW embedding tensors for training & validation
              device: str, n_epochs: int,  # param.s for running model
              n_MLP_layers: int, dim_MLP: int, dim_n2v: int, n_GNN_layers: int, dim_between: int, dim_U: int, n_U_layers: int,  # core param.s
              l: int = 8, n_pass: int = 1, scatter_type: str = 'sum', lr_gnn: float = 0.001  # possible default param.s
              ) -> tuple[float, float] | list[int]:
    '''runs GNN model on node_attributes from given (sparse rep.s of) training & validation graphs together w/ respective, pre-computed (p-trees-)node2vec & closed walk kernel (CW) embedding slices'''
    # construct GNN model w/ given param.s:
    model = GNN(n_MLP_layers, dim_MLP, dim_n2v, n_GNN_layers, dim_between, dim_U, n_U_layers,  # core param.s
                l, n_pass, scatter_type  # possible default param.s
                )
    model.to(device)  # move model to device
    model.train()  # switch model to training mode
    optimizer = Adam(model.parameters(), lr=lr_gnn)  # construct optimizer

    accuracies = []  # collect accuracies over epochs
    for epoch in range(1, n_epochs + 1):  # run training & evaluation phase for [n_epochs]
        out_train = model(G_train, X_train, W_train)  # forward pass on training graph
        #loss = F.cross_entropy(out_train, G_train.node_labels, reduction='mean')
        loss = F.cross_entropy(out_train, G_train.node_labels, reduction='sum')  # training loss
        loss.backward()  # backward pass
        optimizer.step()  # SGD step
        optimizer.zero_grad()  # set gradients to zero
        accuracy_train = accuracy(out_train, G_train.node_labels)  # accuracy on train data themselves
        accuracy_train_round = round(accuracy_train * 100, 4)  # rounded in %

        # validation phase:
        model.eval()  # switch model to evaluation mode
        with th.no_grad():
            out_val = model(G_val, X_val, W_val)  # evaluate forward fct. on validation graph to predict node_labels thereof
            if G_val.set_node_labels:  # cross-validation mode
                accuracy_val = accuracy(out_val, G_val.node_labels)  # accuracy on val data
                # print progress for loss, training accuracy & validation accuracy (rounded):
                print(f"epoch {epoch}:\tloss_train: {loss.item():.4f}\t\tacc_train(%): {accuracy_train_round}\t\tacc_val(%): {accuracy_val * 100:.4f}")
                accuracies.append(accuracy_val)  # add val-accuracy
                if epoch == n_epochs:
                    return_obj = accuracy_val, argmax(accuracies).astype(float)  # return accuracy & best epoch for validation graph
            else:  # prediction mode
                print(f"epoch {epoch}:\tloss_train: {loss.item():.4f}\t\tacc_train(%): {accuracy_train_round}")
                if epoch == n_epochs:
                    return_obj = out_val.argmax(-1).tolist()  # return final node_label prediction (see task)

    return return_obj  # return_object from last epoch



#print_progress = False
if __name__ == "__main__":
    # test model on exemplary train/val split w/ dummy embeddings for X & W:
    from numpy import arange, setdiff1d
    from numpy.random import default_rng
    import pickle
    from timeit import default_timer
    import torch as th
    #from torch.cuda import is_available as cuda_is_available
    #from torch.backends.mps import is_available as mps_is_available
    #print_progress = True

    # parameters:
    k = 12
    #pred_mode = False
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    #device = ("cuda" if cuda_is_available() else "mps" if mps_is_available() else "cpu")
    n_epochs = 20
    n_MLP_layers = 6
    dim_MLP = 50 #30 #50 #150
    dim_n2v = 128
    n_GNN_layers = 3
    dim_between = 50 #15 #30 #50
    dim_U = 50 #15 #30 #50
    n_U_layers = 2
    #p = 0.5  # for p-trees
    #m = 10  # for p-trees
    #m_ns = 10  # for p-trees
    #batch_size = 100  # for node2vec
    #n_batches = 100  # for node2vec
    # param.s w/ given default:
    l = 8  # for CW
    n_pass = 1 #2 #3
    scatter_type = 'sum' #'mean' #'max'
    lr_gnn = 0.001 #0.01 #0.001
    #lr_n2v = 0.01 #0.01 #0.001

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    t_start = default_timer()
    cutoff = 1000
    rng = default_rng(seed=None)

    #G_full = Sparse_Graph(graph, False)  # sparse rep. of full graph
    #n_nodes = G_full.n_nodes
    n_nodes = graph.number_of_nodes()
    n_nodes_learn = n_nodes - cutoff
    n_nodes_val = n_nodes_learn // k  # number of nodes in (regular) validation split/subgraph
    #n_nodes_val = n_nodes // k
    n_nodes_train = n_nodes_learn - n_nodes_val
    nodes_learn = arange(cutoff, n_nodes)  # nodes (indices) of learning subgraph (has node_labels, used for cross-validation)
    dim_cw = l - 1

    # train & val node splits:
    nodes_val = rng.choice(nodes_learn, size=n_nodes_val, replace=False, p=None, axis=0, shuffle=False)  # sample val split from nodes_learn
    nodes_train = setdiff1d(nodes_learn, nodes_val, assume_unique=True)

    # sparse graph rep.s:
    G_train = Sparse_Graph(graph.subgraph(nodes_train), True)
    G_val = Sparse_Graph(graph.subgraph(nodes_val), True)
    """# dummy embeddings (zeros):
    X_train = th.zeros((n_nodes_train, dim_n2v))
    X_val = th.zeros((n_nodes_val, dim_n2v))
    W_train = th.zeros((n_nodes_train, dim_cw))
    W_val = th.zeros((n_nodes_val, dim_cw))
    """# (random):
    X_train = th.normal(0, 1, (n_nodes_train, dim_n2v))
    X_val = th.normal(0, 1, (n_nodes_val, dim_n2v))
    W_train = th.rand((n_nodes_train, dim_cw), dtype=th.float64) #dtype=th.float64
    W_val = th.rand((n_nodes_val, dim_cw), dtype=th.float64)
    #

    result = run_model(G_train, G_val,  # sparse graph rep.s for training & validation
              X_train, X_val,  # node2vec embedding tensors for training & validation
              W_train, W_val,  # CW embedding tensors for training & validation
              device, n_epochs,  # param.s for running model
              n_MLP_layers, dim_MLP, dim_n2v, n_GNN_layers, dim_between, dim_U, n_U_layers,  # core param.s
              l, n_pass, scatter_type, lr_gnn  # possible default param.s
              )
    print(f"\nTime = {(default_timer() - t_start) / 60} mins")
