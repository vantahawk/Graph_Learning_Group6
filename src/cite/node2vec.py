'''implementation of node2vec for p-trees, identical to version for pq-walks except for name changes, computation shortcuts and use of Sparse_Graph'''
# external imports:
#import numpy as np
import torch as th  # TODO replace th. w/ direct imports (maybe)
from torch.nn import Module, Parameter
from torch.nn.init import kaiming_normal_#, kaiming_uniform_, normal_, uniform_, xavier_normal_, xavier_uniform_
from torch.optim import Adam
from torch.utils.data import DataLoader

# internal imports:
#from random_walks import RW_Iterable
from random_trees import RT_Iterable
from sparse_graph import Sparse_Graph



class Node2Vec(Module):
    '''node2vec embedding as torch module w/ embedding matrix X as parameter & the def. stochastic loss fct. as foward fct.'''
    def __init__(self, n_nodes: int, dim_n2v: int, m: int, device: str) -> None:
        super().__init__()

        # main attributes of rt_batch
        self.m = m
        #self.m = th.tensor(m).to(device)  # random walk length
        self.tree_dim = m + 1
        #self.tree_dim = self.m + 1
        #self.sum_loss = th.tensor(0.)  # initialize sum of loss values
        self.device = device

        # initialize node2vec embedding matrix X randomly as parameter:
        self.X = Parameter(th.empty(n_nodes, dim_n2v))  # n_nodes x dim_n2v
        # test different initializations:
        kaiming_normal_(self.X)
        #kaiming_uniform_(self.X)
        #normal_(self.X, mean=0.0, std=1.0)
        #uniform_(self.X, a=-1.0, b=1.0)
        #xavier_normal_(self.X)
        #xavier_uniform_(self.X)


    def forward(self, rt_batch: th.Tensor) -> th.Tensor:
        '''forward fct. of node2vec embedding, takes batch matrix (2D) of stacked p_tree data, returns scalar value of mean loss fct. over p_tree batch (simplified, see conversion) as def. in sheet/script 4'''
        sum_loss = th.tensor(0., dtype=th.float64).to(self.device)  # initialize sum of loss values
        #sum_loss = self.sum_loss

        for rt_vec in rt_batch:  # run over p_tree data vectors in batch
            X_start = self.X[rt_vec[0]]  # embedding vec. of start node (X_s)
            # add loss value for each p_tree (see conversion of loss-fct.), rt_vec[1 : self.tree_dim] = p-tree nodes excl. start node:
            sum_loss += (self.m * th.log(th.sum(th.exp(th.matmul(self.X[rt_vec], X_start)), -1))  # denominator-term
                         - th.sum(th.matmul(self.X[rt_vec[1 : self.tree_dim]], X_start), -1))  # numerator-term
        #batch_loss = self.sum_loss / len(rt_batch)
        #self.sum_loss = 0
        return sum_loss / len(rt_batch)  # return mean loss over batch
        #return th.tensor(batch_loss)



def train_node2vec(dataset: RT_Iterable, dim_n2v: int, m: int,  # main parameters, see sheet
                   n_batches: int, batch_size: int, device: str,  # extra parameters
                   lr_n2v: float = 0.01) -> th.Tensor:  # learning rate  # standard: lr_n2v: float = 0.001  # our default: lr_n2v: float = 0.01
    '''trains [dim_n2v]-dimensional node2vec model on given graph w/ Adam optimizer, using [batch_size] random trees w/ parameters p, m & m_ns'''
    # prepare dataloader (converts np.ndarray rt_batches into th.tensor), model & optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # single-process  # TODO adapt to multi-process slu.
    model = Node2Vec(dataset.n_nodes, dim_n2v, m, device)  # construct model object
    model.to(device)  # move model to device
    model.train()  # switch model to training mode
    optimizer = Adam(model.parameters(), lr=lr_n2v)  # construct optimizer

    # stream [n_batches] random p-tree batches from custom iterable dataset & train model on them successively
    print("Completed p-tree batches: ", end="")
    for i in range(1, n_batches + 1):
        for rt_batch in dataloader:  # access single batch in dataloader
            rt_batch = rt_batch.to(device)  # move random tree batch to device
            loss = model(rt_batch)  # forward pass = compute batch-lvl mean loss
            loss.backward()  # backward pass
            optimizer.step()  # SGD step
            optimizer.zero_grad()  # set gradients to zero
        print(f"{i}, ", end="")

    print("\n")
    model.eval()  # switch model to evaluation mode (optional?)
    return model.get_parameter('X').detach().type(th.float64)  # return embedding matrix X as th.tensor
    #return model.get_parameter('X').detach().numpy()  # return embedding matrix X as np.ndarray
    #return model.get_parameter('X').detach().to('cpu').numpy()  # move back to CPU first?
    #return model.get_parameter('X').numpy(force=True)  # forced version



if __name__ == "__main__":
    # test node2vec embedding:
    import pickle
    from timeit import default_timer
    #from torch.cuda import is_available as cuda_is_available
    #from torch.backends.mps import is_available as mps_is_available

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 10 #10 #100 #1000
    n_batches = 10
    dim_n2v = 128 #12
    p = 0.5 #0.1 #0.2 #0.5 #0.8 #1.0
    #q = 1.0
    m = 10 #10 #20 #50 #100
    m_ns = 10 #10 #20 #50 #100
    set_node_labels = False

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    t_start = default_timer()
    dataset = RT_Iterable(Sparse_Graph(graph, set_node_labels), p, m, m_ns, batch_size)  # custom iterable dataset instance
    X = train_node2vec(dataset, dim_n2v, m, n_batches, batch_size, device)
    print(f"Time = {(default_timer() - t_start) / 60} mins\n{X[0]}")
