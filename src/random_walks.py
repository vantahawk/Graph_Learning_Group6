from networkx import Graph, adjacency_matrix, number_of_edges, number_of_nodes
import numpy as np
from numpy.random import default_rng
from torch.utils.data import IterableDataset#, get_worker_info
from typing import Iterator



class RW_Iterable(IterableDataset):
    '''implements custom iterable dataset for random [pq]-walks of length l over given [graph], together w/ [l_ns] negative samples'''
    def __init__(self, graph: Graph, p: float, q: float, l: int, l_ns: int,  # main parameters, see sheet
                 batch_size: int, set_node_labels: bool) -> None:  # extra parameters
        super().__init__()

        # main graph attributes
        self.adj_mat = adjacency_matrix(graph).toarray()  # int32-np.ndarray
        self.n_nodes = number_of_nodes(graph)

        if set_node_labels:  # for node classification (Ex.3)
            # 1D-np.ndarray, size: n_nodes, fails for Facebook, unknown why, not needed tho
            self.node_labels = np.array([node[1]['node_label'] for node in graph.nodes(data=True)])
        else:  # only set edges instead, for link prediction (Ex.4)
            self.n_edges = number_of_edges(graph)
            # 2D-np.ndarray of (unique) edges, size: n_edges x 2, subtract 1 elem.wise to account for node count starting at zero
            self.edges = np.array([[edge[0], edge[1]] for edge in graph.edges(data=True)]) - 1

        # attributes for random walk generation
        self.rng = default_rng(seed=None)  # default RNG object for general random sampling
        self.p = p
        self.q = q
        self.l = l  # random walk length (excl. start node)
        self.walk_dim = self.l + 1  # (incl. start node)
        self.l_ns = l_ns  # number of negative samples
        self.batch_size = batch_size  # size of random walk batch generated by rw_batch()


    def rw_batch(self) -> list[np.ndarray]:
        '''returns batch (list) of pq-walk data, each including: random start node, followed by l nodes of random pq-walk, followed by l_ns negative samples, concatenated into 1D-np.ndarray'''
        batch = np.zeros((self.batch_size, self.walk_dim + self.l_ns), dtype=np.int32)  # ...in np.array

        for walk in range(self.batch_size):
            # last node, initially the uniformly sampled start node
            last = self.rng.choice(self.n_nodes, size=None, replace=True, p=None, axis=0, shuffle=True)  # np.int32
            start_nbh = self.adj_mat[last]  # neighborhood of start node repres. as resp. row of adjacency matrix
            # current node, initially the 2nd node of pq-walk, uniformly sampled from neighborhood of start node
            current = self.rng.choice(self.n_nodes, size=None, replace=True, p = start_nbh / np.sum(start_nbh), axis=0, shuffle=True)
            # collect sampled nodes of pq-walk array
            pq_walk = np.zeros(self.walk_dim, dtype=np.int32)
            pq_walk[0], pq_walk[1] = last, current

            # sample the l-1 next nodes in pq-walk using algebraic construction of alpha (see def. in script/sheet)
            for node in range(2, self.walk_dim):
                current_nbh = self.adj_mat[current]  # neighborhood of current node repres. as its adj.mat.row
                # common neighborhood of last & current node, repres. as elem-wise product of resp. adj.mat.rows, accounts for 2nd row in def. of alpha
                #common_nbh = np.multiply(self.adj_mat[last], current_nbh)
                common_nbh = self.adj_mat[last] * current_nbh
                # alpha repres. as array of disc. probab.s over all nodes (not norm.ed), hence the use of adj.mat.rows to represent neighborhoods
                alpha = common_nbh + (current_nbh - common_nbh) / self.q  # accounts for 1st & 2nd row in def. of alpha
                alpha[last] = 1 / self.p  # accounts for 3rd row in def. of alpha (step back to last node)

                # sample next node in pq-walk according to norm.ed alpha (discrete probab. over all nodes)
                next = self.rng.choice(self.n_nodes, size=None, replace=True, p = alpha / np.sum(alpha), axis=0, shuffle=True)
                pq_walk[node] = next

                # update last & current node
                last = current
                current = next

            rest_nodes = np.delete(np.arange(self.n_nodes), pq_walk)  # remaining nodes after drawn pq-walk

            # negative samples (np.ndarray) uniformly drawn from remaining nodes
            # FIXME whether to draw negative samples w/ or w/o repetition
            neg_samples = self.rng.choice(rest_nodes, size=self.l_ns, replace=False, p=None, axis=0, shuffle=False)  # w/o repetition
            #neg_samples = self.rng.choice(rest_nodes, size=self.l_ns, replace=True, p=None, axis=0, shuffle=False)  # w/ repetition
            batch[walk] = np.concatenate([pq_walk, neg_samples], axis=-1)  # concatenate walk & negative samples into pq-walk data vector

        return list(batch)  # dataset supposed to be same size as batch here, gets cast to th.tensor by dataloader


    def __iter__(self) -> Iterator[np.ndarray]:
        '''returns iterator of pq-walk data, single-process: runs slow, unreliable or unintended otherwise...'''
        return iter(self.rw_batch())



if __name__ == "__main__":
    # for testing streaming, batching, multiprocessing, etc.
    import pickle
    from psutil import cpu_count
    from torch.utils.data import DataLoader, get_worker_info

    #n_workers = 2 #cpu_count(logical=True)
    batch_size =  5 #3 * n_workers

    with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    dataset = RW_Iterable(graph, p=1.0, q=1.0, l=5, l_ns=5, batch_size=batch_size, set_node_labels=False)  # custom iterable dataset instance
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # turns random walk batches into th.tensors, single-process

    # multi-process part removed
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, worker_init_fn=worker_split)  # issue w/ worker_split(), produces failed subprocesses
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)  # works but returns one batch for each subprocess? also duplicates depending on implementation (expected but not efficiently fixed yet)

    for run in range(3):
        for batch in dataloader:
            print(batch)
        print("\n")
