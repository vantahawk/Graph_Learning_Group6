# external imports
#import networkx as nx
import numpy as np
import torch as th
from torch.nn import Module, Parameter
#import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, kaiming_uniform_, normal_, uniform_, xavier_normal_, xavier_uniform_
from torch.optim import Adam
from torch.utils.data import DataLoader

# internal imports
from random_walks import RW_Iterable



class Node2Vec(Module):
    '''node2vec embedding as torch module w/ embedding matrix X as parameter & the def. stochastic loss fct. as foward fct.'''
    def __init__(self, n_nodes: int, dim: int, l: int) -> None:
        super().__init__()

        # main attributerw_batch
        #self.n_nodes = n_nodes  # number of nodes
        #self.dim = dim  # embedding dimension
        self.l = l  # random walk length

        # initialize node2vec embedding matrix X randomly as parameter
        self.X = Parameter(th.empty(n_nodes, dim))  # n_nodes x dim
        # TODO test different initializations?
        kaiming_normal_(self.X)
        #kaiming_uniform_(self.X)
        #normal_(self.X, mean=0.0, std=1.0)
        #uniform_(self.X, a=-1.0, b=1.0)
        #xavier_normal_(self.X)
        #xavier_uniform_(self.X)


    def forward(self, rw_batch: th.Tensor) -> th.Tensor:
        '''forward fct. of node2vec embedding, takes batch matrix (2D) of stacked pq-walk data, returns scalar value of mean loss fct. over pq-walk batch (simplified, see conversion) as def. in sheet/script'''
        sum_loss = th.tensor(0.)  # initialize sum of loss values
        for rw_vec in list(rw_batch):  # run over pq-walk data vectors in batch
            X_start = self.X[rw_vec[0]]  # embedding vec. of start node (X_s)
            walk_idx = rw_vec[: self.l + 1]  # selection from X acc. to pq-walk nodes, including start node
            neg_idx = rw_vec[self.l + 1 :]  # selection from X acc. to negative samples
            numerator_term = th.sum(th.matmul(self.X[walk_idx[1 :]], X_start), -1)  # see conversion of loss-fct., using walk_idx w/o start node

            # FIXME whether to reassign walk_idx to only include each node once (i.e. interpret pq-walk "w" as set rather than sequence in denominator term), interpretation in script/sheet unclear, (de)comment the following line accordingly:
            walk_idx = th.tensor(list(set(np.array(walk_idx))))

            # add loss value for each pq-walk: compute denominator term (see conversion of loss-fct.), then subtract numerator term
            sum_loss += self.l * th.log(th.sum(th.exp(th.matmul(self.X[th.cat([walk_idx, neg_idx], -1)], X_start)), -1)) - numerator_term

        return sum_loss / len(rw_batch)  # return mean loss over batch

"""
    def forward(self, rw_batch: th.Tensor) -> th.Tensor:
        '''short version of forward fct. of node2vec embedding, interpreting pq-walk "w" as sequence rather than set within denominator term'''
        sum_loss = th.tensor(0.)  # initialize sum of loss values
        for rw_vec in list(rw_batch):  # run over pq-walk data vectors in batch
            X_start = self.X[rw_vec[0]]  # embedding vec. of start node (X_s)
            # add loss value for each pq-walk all in one go
            sum_loss += self.l * th.log(th.sum(th.exp(th.matmul(self.X[rw_vec], X_start)), -1)) - th.sum(th.matmul(self.X[rw_vec[1 : self.l + 1]], X_start), -1)
        return sum_loss / len(rw_batch)  # return mean loss over batch
"""


def train_node2vec(dataset: RW_Iterable, dim: int, l: int,  # main parameters, see sheet
                   n_batches: int, batch_size: int, device: str,  # extra parameters
                   lr: float = 0.001) -> np.ndarray:  # default learning rate
    '''trains node2vec model on given graph w/ Adam optimizer, using [batch_size] random walks w/ parameters p, q, l, l_ns & embedding [dim]'''
    # prepare dataloader, model & optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # single-process
    model = Node2Vec(dataset.n_nodes, dim, l)  # construct model object
    model.to(device)  # move model to device
    model.train()  # switch model to training mode
    optimizer = Adam(model.parameters(), lr=lr)  # construct optimizer

    # stream [n_batches] random pq-walk batches W from custom iterable dataset & train model on them successively
    for i in range(n_batches):
        for rw_batch in dataloader:  # access single batch in dataloader
            rw_batch = rw_batch.to(device)  # move random walk batch to device
            #loss = th.mean(model(W), 0)  # forward pass = compute loss, take batch-lvl mean (see sheet/script)
            loss = model(rw_batch)  # # forward pass = compute batch-lvl mean loss
            loss.backward()  # backward pass
            optimizer.step()  # SGD step
            optimizer.zero_grad()  # set gradients to zero

    model.eval()  # switch model to evaluation mode (optional?)
    return model.get_parameter('X').detach().numpy()  # return embedding matrix X as np.ndarray
    #return model.get_parameter('X').detach().to('cpu').numpy()  # move back to CPU first?



if __name__ == "__main__":
    # test node2vec embedding
    import pickle

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 10
    n_batches = 100
    dim = 128 #12
    p = 1.0
    q = 1.0
    l = 5
    l_ns = 5
    set_node_labels = False

    with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels)  # custom iterable dataset instance
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # turns random walk batches into th.tensors, single-process w/ renewing randomness
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device)
    print(X[0])
