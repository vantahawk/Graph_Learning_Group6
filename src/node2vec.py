# external imports
import networkx as nx
import numpy as np
import torch as th
from torch.nn import Module, Parameter
#import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, kaiming_uniform_, normal_, uniform_, xavier_normal_, xavier_uniform_
from torch.optim import Adam
from torch.utils.data import DataLoader

# internal imports
from random_walks import RW_Iterable

import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta:float=0.0001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.best_loss = val_loss


class Node2Vec(Module):
    '''node2vec embedding as torch module w/ embedding matrix X as parameter & the def. stochastic loss fct. as foward fct.'''
    def __init__(self, n_nodes: int, dim: int, l: int) -> None:
        """initializes node2vec embedding as torch module w/ embedding matrix X as parameter & the def. stochastic loss fct. as foward fct.
    
        Args:
            n_nodes (int): number of nodes
            dim (int): embedding dimension
            l (int): random walk length
        """
        super().__init__()

        self.l = l  # random walk length

        # initialize node2vec embedding matrix X randomly as parameter
        self.X = Parameter(th.empty(n_nodes, dim))  # n_nodes x dim
        kaiming_normal_(self.X)


    def forward(self, rw_batch: th.Tensor) -> th.Tensor:
        '''forward fct. of node2vec embedding, takes batch matrix (2D) of stacked pq-walk data, returns scalar value of mean loss fct. over pq-walk batch (simplified, see conversion) as def. in sheet/script
        
        Args:
            rw_batch (th.Tensor): batch matrix (2D) of stacked pq-walk data: [batch_size, l+1+l_ns] (l+1 for pq-walk, l_ns for negative samples)
            
        Returns:
            th.Tensor: scalar value of mean loss fct. over pq-walk batch (simplified, see conversion) as def. in sheet/script'''

        #now get rid of the for loop, by doing it in parallel
        X_start = self.X[rw_batch[:, 0]]  # embedding vec. of start node (X_s)
        walk_idx = rw_batch[:, :self.l+1]
        neg_idx = rw_batch[:, self.l+1:]

        numerator_term = th.sum(th.bmm(self.X[walk_idx[:, 1:]], X_start.unsqueeze(2)), 1)

        # add loss value for each pq-walk: compute denominator term (see conversion of loss-fct.)
        loss = self.l * th.log(th.sum(th.exp(th.bmm(self.X[th.cat([walk_idx.unique(dim=1), neg_idx], 1)], X_start.unsqueeze(2))), 1)) - numerator_term
        #above is actually this: -log(exp numerator/sum exp denominator) = -log(exp numerator) + log(sum exp denominator) = -numerator + log(sum exp denominator)

        return loss.mean()


#THESE ARE THE OVERLOADS, USED FOR TYPE HINTING 
from typing import Generator, overload, Literal
@overload
def train_node2vec(dataset: RW_Iterable, dim: int, l: int,
                    n_batches: int, batch_size: int, device: str,
                    lr: float = 0.001, lrsched:str="constant", delta:float=0.01, verbose:bool=False, yield_X:Literal[False]=False) -> np.ndarray:
    """Trains node2vec model on given graph w/ Adam optimizer, using [batch_size] random walks w/ parameters p, q, l, l_ns & embedding [dim]

    Returns embedding matrix X as np.ndarray"""
    ...

@overload
def train_node2vec(dataset: RW_Iterable, dim: int, l: int,
                    n_batches: int, batch_size: int, device: str,
                    lr: float = 0.001, lrsched:str="constant", delta:float=0.01, verbose:bool=False, yield_X:Literal[True]=True) -> Generator:
    """Trains node2vec model on given graph w/ Adam optimizer, using [batch_size] random walks w/ parameters p, q, l, l_ns & embedding [dim]
    
    Returns embedding matrix X as np.ndarray, yields X at each epoch if [yield_X] is True"""
    ...

def train_node2vec(dataset: RW_Iterable, dim: int, l: int,  # main parameters, see sheet
                   n_batches: int, batch_size: int, device: str,  # extra parameters
                   lr: float = 0.001, lrsched:str="constant", delta:float=0.01, verbose:bool=False, yield_X:bool=False):  # default learning rate
    '''trains node2vec model on given graph w/ Adam optimizer, using [batch_size] random walks w/ parameters p, q, l, l_ns & embedding [dim]
    
    returns embedding matrix X as np.ndarray, optionally yields X at each epoch if [yield_X] is True'''
    # prepare dataloader, model & optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # single-process
    model = Node2Vec(dataset.n_nodes, dim, l)  # construct model object
    model.to(device)  # move model to device
    model.train()  # switch model to training mode
    optimizer = Adam(model.parameters(), lr=lr)  # construct optimizer
    #if no constant learnong rate, construct scheduler
    if lrsched != "constant":
        lrsched = {
            "cosine":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_batches),  # cosine
            "plateau":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=verbose),
            "step":torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, verbose=verbose),
            "linear":torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=n_batches, verbose=verbose),
        }[lrsched]

    #early stopping
    early_stopping = EarlyStopping(patience=10, verbose=verbose, delta=delta)
    
    # doing it this way, we can yield the X at each epoch
    # then we can train logReg-Classifier on each epoch, this gives nice visualizations
    def train():
        while(True):
            nonlocal n_batches
            n_batches -= 1
            if n_batches < 0:
                break
            model.train()
            losses = []
            for rw_batch in dataloader:  # access single batch in dataloader
                rw_batch = rw_batch.to(device)  # move random walk batch to device
                #loss = th.mean(model(W), 0)  # forward pass = compute loss, take batch-lvl mean (see sheet/script)
                loss = model(rw_batch)  # # forward pass = compute batch-lvl mean loss
                losses.append(loss)
                loss.backward()  # backward pass
                optimizer.step()  # SGD step
                optimizer.zero_grad()  # set gradients to zero
            mean_loss = th.mean(th.stack(losses))
            # if not constant learning rate, step scheduler
            if lrsched != "constant":
                if not isinstance(lrsched,torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lrsched.step()
                else:
                    lrsched.step(mean_loss)
            early_stopping(mean_loss, model)
            if early_stopping.early_stop:
                break

            model.eval()
            yield model.get_parameter('X').detach().numpy()

    if yield_X:
        return train()
    else:
        for _ in train():
            pass
        model.eval()  # switch model to evaluation mode (optional?)
        return model.get_parameter('X').detach().numpy()  # return embedding matrix X as np.ndarray


if __name__ == "__main__":
    # test node2vec embedding
    import pickle

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 10
    n_batches = 100
    dim = 12
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
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device, lr=0.01, lrsched="cosine", delta=0.01, verbose=True, yield_X=False)
    print(X[0])
