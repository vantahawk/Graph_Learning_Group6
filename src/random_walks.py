from networkx import Graph, adjacency_matrix, number_of_edges, number_of_nodes
import networkx as nx
#from numpy import ndarray, array, concatenate, sum
import numpy as np
from numpy.random import default_rng
#import torch as th
from torch.utils.data import IterableDataset#, get_worker_info
from typing import Iterator

import torch.multiprocessing as mp
from multiprocessing.pool import Pool
import time
from itertools import repeat

#import the buffer type from numpy
from collections.abc import Buffer


"""
def one_hot_encoder(label: int, length: int) -> np.ndarray:
    '''returns one-hot vector according to given label integer and length'''
    zero_vector = np.zeros(length)
    zero_vector[label] = 1
    return zero_vector
"""
# Global variables to be used by the worker processes
global_shared_adj:Buffer|None = None
worker_pool: Pool|None = None

def init_worker(shared_adj):
    """Initializes the shared adjacency matrix for the worker processes."""
    global global_shared_adj
    global_shared_adj = shared_adj

class RW_Iterable(IterableDataset):
    '''implements custom iterable dataset for random [pq]-walks of length l over given [graph], together w/ [l_ns] negative samples'''
    def __init__(self, graph: nx.Graph, p: float, q: float, l: int, l_ns: int,  # main parameters, see sheet
                 batch_size: int, set_node_labels: bool, n_workers:int=-1) -> None:  # extra parameters
        super().__init__()
        # main graph attributes
        #self.graph = graph  # full nx.Graph
        adj_mat:np.ndarray = nx.adjacency_matrix(graph).toarray()  # int32-np.ndarray
        #self.adj_mat = th.tensor(nx.adjacency_matrix(graph).toarray())  # int32-th.tensor
        self.n_nodes:int = nx.number_of_nodes(graph)

        if set_node_labels:  # for node classification (Ex.3)
            # 1D-np.ndarray, size: n_nodes, fails for Facebook, unknown why, not needed tho
            self.node_labels = np.array([node[1]['node_label'] for node in graph.nodes(data=True)])
        else:  # only set edges instead, for link prediction (Ex.4)
            self.n_edges = number_of_edges(graph)
            # 2D-np.ndarray, size: n_edges x 2, subtract 1 elem.wise to account for node count starting at zero
            self.edges = np.array([[edge[0], edge[1]] for edge in graph.edges(data=True)]) - 1
            #self.edges = th.tensor(self.edges)  # th.tensor

        # attributes for random walk generation
        # self.rng = default_rng(seed=None)  # default RNG object for general random sampling
        self.p = p
        self.q = q
        self.l = l  # random walk length (excl. start node)
        self.l_ns = l_ns  # number of negative samples

        # attributes for batching & worker split
        self.batch_size = batch_size  # size of random walk batch generated by rw_batch()
        self.num_workers = mp.cpu_count() if n_workers <=0 else n_workers  # number of worker processes for multiprocessing
        shared_adj = mp.RawArray('i', adj_mat.flatten())  #read-only shared memory for adjacency matrix
        self.adj_shape = adj_mat.shape
        global worker_pool
        worker_pool = mp.Pool(self.num_workers, initializer=init_worker, initargs=(shared_adj, ) )  # pool of worker processes for multiprocessing
        #self.start = 0
        #self.end = batch_size
        self.sampled_walks = 0

    def rw_batch(self) -> list[np.ndarray]:
        '''returns batch (list) of pq-walk data, each including: random start node, followed by l nodes of random pq-walk, followed by l_ns negative samples, concatenated into 1D-np.ndarray'''

        #submit tasks to the pool
        global worker_pool
        if worker_pool is None:
            raise RuntimeError("Pool not initialized")
        b = self.batch_size
        self.sampled_walks += b #count the number of sampled walks, so that seed can be different for each batch
        return worker_pool.starmap(self.random_walk, zip(repeat(self.n_nodes,b), repeat(self.adj_shape,b), repeat(self.l,b), repeat(self.l_ns,b), repeat(self.p,b), repeat(self.q,b), range(self.sampled_walks-b, self.sampled_walks)))

    @staticmethod
    def random_walk(n_nodes:int, adj_shape, l, l_ns, p, q, seed:int)->np.ndarray:
        """returns pq-walk data array, including: random start node, followed by l nodes of random pq-walk, followed by l_ns negative samples, concatenated into 1D-np.ndarray

        Args:
            n_nodes (int): number of nodes in the graph
            adj_shape (tuple): shape of the adjacency matrix
            l (int): length of the pq-walk
            l_ns (int): number of negative samples
            p (float): return parameter
            q (float): in-out parameter
            seed (int): seed for the random number generator

        Returns:
            np.ndarray: pq-walk data array, consisting of the random start node, followed by l nodes of random pq-walk, followed by l_ns negative samples"""
        # last node, initially the uniformly sampled start node
        #last = self.rng.integers(0, high=self.n_nodes, size=None, dtype=np.int64, endpoint=False)
        rng = default_rng(seed)
        last = rng.choice(n_nodes, size=None, replace=True, p=None, axis=0, shuffle=True)  # np.int32

        #get the shared adjacency matrix
        global global_shared_adj
        if global_shared_adj is None:
            raise RuntimeError("Shared adjacency matrix not initialized")
        adj_mat = np.frombuffer(global_shared_adj, dtype=np.int32).reshape(adj_shape)  # np.ndarray

        start_nbh = adj_mat[last]  # neighborhood of start node repres. as resp. row of adjacency matrix
        # current node, initially the 2nd node of pq-walk, uniformly sampled from neighborhood of start node
        current = rng.choice(n_nodes, size=None, replace=True, p = start_nbh / np.sum(start_nbh), axis=0, shuffle=True)

        #pq_walk = np.zeros((l,), dtype=np.int32)
        pq_walk = np.zeros((l + 1,), dtype=np.int32)
        pq_walk[0] = last
        pq_walk[1] = current

        #for step in range(2, l):  # sample the l-1 next nodes in pq-walk using algebraic construction of alpha (see def. in script/sheet)
        for step in range(2, l + 1):
            current_nbh = adj_mat[current]  # neighborhood of current node repres. as its adj.mat.row
            # common neighborhood of last & current node, repres. as elem-wise product of resp. adj.mat.rows, accounts for 2nd row in def. of alpha
            common_nbh = np.multiply(adj_mat[last], current_nbh)
            # alpha repres. as array of disc. probab.s over all nodes (not norm.ed), hence the use of adj.mat.rows to represent neighborhoods
            alpha = common_nbh + (current_nbh - common_nbh) / q  # accounts for 1st & 2nd row in def. of alpha
            alpha[last] = 1 / p  # accounts for 3rd row in def. of alpha (step back to last node)

            # sample next node in pq-walk according to norm.ed alpha (discrete probab. over all nodes)
            next = rng.choice(n_nodes, size=None, replace=True, p = alpha / np.sum(alpha), axis=0, shuffle=True)
            pq_walk[step] = next

            # update last & current node
            last = current
            current = next

        rest_nodes = np.arange(n_nodes)  # remaining nodes after drawn pq-walk
        rest_nodes = np.delete(rest_nodes, pq_walk)  # remove nodes already in pq-walk
        # negative samples (np.ndarray) uniformly drawn from remaining nodes
        neg_samples = rng.choice(rest_nodes, size=l_ns, replace=False, axis=0, shuffle=False)  # w/o repetition

        return np.concatenate([pq_walk, neg_samples], axis=-1)  # gets cast to th.tensor by dataloader

    def __iter__(self) -> Iterator[np.ndarray]:
        '''returns iterator of pq-walk data'''
        return iter(self.rw_batch())

    def get(self)->np.ndarray:
        return np.array(self.rw_batch())



if __name__ == "__main__":
    # for testing streaming, batching, multiprocessing, etc.
    import pickle
    from psutil import cpu_count
    from torch.utils.data import DataLoader, get_worker_info
    """
    def worker_split(worker_id) -> None:
        worker_info = get_worker_info()
        ds = worker_info.dataset  # the dataset copy in this worker process
        overall_start = ds.start
        overall_end = ds.end
        # configure the dataset to only process the split workload
        per_worker = int(np.ceil((overall_end - overall_start) / worker_info.num_workers))
        #worker_id = worker_info.id
        ds.start = overall_start + worker_info.id * per_worker
        #ds.start = overall_start + worker_id * per_worker
        ds.end = min(ds.start + per_worker, overall_end)
    """
    batch_size = 300 #3 * n_workers

    with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    dataset = RW_Iterable(graph, p=1.0, q=1.0, l=20, l_ns=20, batch_size=batch_size, set_node_labels=False, n_workers=1)  # custom iterable dataset instance
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # turns random walk batches into th.tensors, single-process w/ renewing randomness

    from timeit import default_timer as timer

    start = timer()
    for run in range(3):
        for batch in dataloader:
            print(batch)
        print("\n")
    if not worker_pool is None:
        worker_pool.close()
        worker_pool.join()
    end = timer()
    print(f"Time elapsed: {end - start:.2f} s")
